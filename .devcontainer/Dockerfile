FROM ghcr.io/prefix-dev/pixi:0.45.0-bullseye-slim

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.pixi-env \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

WORKDIR /workspaces/joshpy
USER $USERNAME
CMD ["bash"]