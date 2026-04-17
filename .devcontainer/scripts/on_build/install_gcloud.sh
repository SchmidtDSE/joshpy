#!/bin/bash
# License: BSD-3-Clause
#
# SHA256 captured 2026-04-16 from this devcontainer (Codespaces, linux/amd64).
# Source: https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-526.0.0-linux-x86_64.tar.gz
# To update: download the new version, run sha256sum, and replace both
# the version and hash below.
set -e

GCLOUD_VERSION="526.0.0"
GCLOUD_SHA256="9d647a35c87e3d6ffe3f0c7331a81b3c7cd02b0bd1cb48b83f6acb5aca75d000"

curl -fsSL "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-linux-x86_64.tar.gz" -o /tmp/gcloud.tar.gz

if ! echo "${GCLOUD_SHA256}  /tmp/gcloud.tar.gz" | sha256sum --check --status; then
  echo "ERROR: SHA256 verification failed for google-cloud-cli ${GCLOUD_VERSION}." >&2
  echo "Expected: ${GCLOUD_SHA256}" >&2
  echo "Got:      $(sha256sum /tmp/gcloud.tar.gz | cut -d' ' -f1)" >&2
  echo "The upstream archive may have changed. Re-verify and update the hash." >&2
  rm /tmp/gcloud.tar.gz
  exit 1
fi

tar -xzf /tmp/gcloud.tar.gz -C /opt
/opt/google-cloud-sdk/install.sh --quiet --usage-reporting=false --command-completion=false --path-update=false
/opt/google-cloud-sdk/bin/gcloud components install gke-gcloud-auth-plugin --quiet
rm /tmp/gcloud.tar.gz

# Symlink to /usr/local/bin so gcloud and the auth plugin are on the system
# PATH for all processes — not just interactive shells. kubectl exec's
# gke-gcloud-auth-plugin as a subprocess, which in turn exec's gcloud;
# neither inherits shell PATH or Dockerfile ENV PATH reliably.
ln -sf /opt/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud
ln -sf /opt/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil
ln -sf /opt/google-cloud-sdk/bin/gke-gcloud-auth-plugin /usr/local/bin/gke-gcloud-auth-plugin
