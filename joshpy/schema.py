# SQL Schema
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sweep_sessions (
    session_id      VARCHAR PRIMARY KEY,
    experiment_name VARCHAR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    template_path   VARCHAR,
    template_hash   VARCHAR(12),
    simulation      VARCHAR,
    total_jobs      INTEGER,
    total_replicates INTEGER,
    status          VARCHAR DEFAULT 'pending',
    metadata        JSON
);

CREATE TABLE IF NOT EXISTS job_configs (
    run_hash        VARCHAR(12) PRIMARY KEY,
    session_id      VARCHAR REFERENCES sweep_sessions(session_id),
    josh_path       TEXT,
    josh_content    TEXT,
    config_content  TEXT,
    file_mappings   JSON,
    label           VARCHAR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Lifecycle status: closed enum {active, superseded, bad}. NULL == 'active'
    -- (read via coalesce(status,'active')). Drives default read filtering and
    -- supersession provenance. See REGISTRY_PROVENANCE.md.
    status            VARCHAR,
    superseded_by     VARCHAR(12),
    status_reason     TEXT,
    status_updated_at TIMESTAMP
);


CREATE TABLE IF NOT EXISTS session_configs (
    session_id  VARCHAR REFERENCES sweep_sessions(session_id),
    run_hash    VARCHAR(12) REFERENCES job_configs(run_hash),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, run_hash)
);

CREATE TABLE IF NOT EXISTS config_parameters (
    run_hash        VARCHAR(12) PRIMARY KEY REFERENCES job_configs(run_hash)
);

CREATE TABLE IF NOT EXISTS job_runs (
    run_id          VARCHAR PRIMARY KEY,
    run_hash        VARCHAR(12) REFERENCES job_configs(run_hash),
    session_id      VARCHAR REFERENCES sweep_sessions(session_id),
    replicate       INTEGER,
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    exit_code       INTEGER,
    output_path     VARCHAR,
    error_message   TEXT,
    metadata        JSON
);

CREATE TABLE IF NOT EXISTS run_outputs (
    output_id       VARCHAR PRIMARY KEY,
    run_id          VARCHAR REFERENCES job_runs(run_id),
    output_type     VARCHAR,
    file_path       VARCHAR,
    file_size       BIGINT,
    row_count       INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE SEQUENCE IF NOT EXISTS cell_id_seq START 1;

CREATE TABLE IF NOT EXISTS cell_data (
    cell_id         BIGINT PRIMARY KEY DEFAULT nextval('cell_id_seq'),
    run_id          VARCHAR REFERENCES job_runs(run_id),
    run_hash        VARCHAR(12),
    step            INTEGER NOT NULL,
    replicate       INTEGER NOT NULL,
    position_x      DOUBLE,
    position_y      DOUBLE,
    longitude       DOUBLE,
    latitude        DOUBLE,
    entity_type     VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_cell_run ON cell_data(run_id);
CREATE INDEX IF NOT EXISTS idx_cell_run_hash ON cell_data(run_hash);
CREATE INDEX IF NOT EXISTS idx_cell_step ON cell_data(step);
CREATE INDEX IF NOT EXISTS idx_cell_replicate ON cell_data(replicate);
CREATE INDEX IF NOT EXISTS idx_cell_spatial ON cell_data(longitude, latitude);
CREATE INDEX IF NOT EXISTS idx_cell_step_replicate ON cell_data(step, replicate);

CREATE TABLE IF NOT EXISTS run_tags (
    scope           VARCHAR NOT NULL,
    key             VARCHAR NOT NULL,
    tags            JSON NOT NULL,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (scope, key)
);
"""

# "Current" cell_data: rows belonging to active runs only. "Current" collapses
# entirely into status='active' (NULL read as active). Non-materialized, so there
# is no refresh cost. Created after the status columns are guaranteed to exist
# (see RunRegistry._migrate_schema), since older databases predate them and the
# view references status. See REGISTRY_PROVENANCE.md.
CELL_DATA_CURRENT_VIEW_SQL = """
CREATE OR REPLACE VIEW cell_data_current AS
    SELECT c.*
    FROM cell_data c
    JOIN job_configs j USING (run_hash)
    WHERE coalesce(j.status, 'active') = 'active';
"""
