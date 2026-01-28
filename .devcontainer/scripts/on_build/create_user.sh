#!/bin/bash
set -e

USERNAME="${1:?Username required}"
USER_UID="${2:?UID required}"
USER_GID="${3:?GID required}"

groupadd --gid "$USER_GID" "$USERNAME"
useradd --uid "$USER_UID" --gid "$USER_GID" -m -s /bin/bash "$USERNAME"
echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/"$USERNAME"
chmod 0440 /etc/sudoers.d/"$USERNAME"