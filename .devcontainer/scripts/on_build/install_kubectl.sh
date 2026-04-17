#!/bin/bash
# License: BSD-3-Clause
#
# SHA256 captured 2026-04-16 from this devcontainer (Codespaces, linux/amd64).
# Source: https://dl.k8s.io/release/v1.31.4/bin/linux/amd64/kubectl
# To update: download the new version, run sha256sum, and replace both
# the version and hash below.
set -e

KUBECTL_VERSION="v1.31.4"
KUBECTL_SHA256="298e19e9c6c17199011404278f0ff8168a7eca4217edad9097af577023a5620f"

curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" -o /tmp/kubectl

if ! echo "${KUBECTL_SHA256}  /tmp/kubectl" | sha256sum --check --status; then
  echo "ERROR: SHA256 verification failed for kubectl ${KUBECTL_VERSION}." >&2
  echo "Expected: ${KUBECTL_SHA256}" >&2
  echo "Got:      $(sha256sum /tmp/kubectl | cut -d' ' -f1)" >&2
  echo "The upstream binary may have changed. Re-verify and update the hash." >&2
  rm /tmp/kubectl
  exit 1
fi

install -o root -g root -m 0755 /tmp/kubectl /usr/local/bin/kubectl
rm /tmp/kubectl
