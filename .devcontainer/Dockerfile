FROM ghcr.io/prefix-dev/pixi:0.66.0-bullseye-slim@sha256:277abdf0a164c5c83ab8c68dee0ce48638a56a8a5097bd02c20cdd59e9ac66e3

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.pixi-env \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

# gcloud SDK + kubectl for batch remote targets (K8s, Cloud Run)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/scripts/on_build/install_gcloud.sh /tmp/install_gcloud.sh
COPY .devcontainer/scripts/on_build/install_kubectl.sh /tmp/install_kubectl.sh
RUN chmod +x /tmp/install_gcloud.sh /tmp/install_kubectl.sh \
    && /tmp/install_gcloud.sh \
    && /tmp/install_kubectl.sh \
    && rm /tmp/install_gcloud.sh /tmp/install_kubectl.sh

WORKDIR /workspaces/joshpy
USER $USERNAME
CMD ["bash"]