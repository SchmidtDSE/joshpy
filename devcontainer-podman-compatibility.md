# Dev Containers: Podman Compatibility Issues

This document describes known compatibility issues when using VS Code Dev Containers with Podman instead of Docker, particularly in rootless mode on Linux.

## Background

VS Code's Dev Containers extension was designed primarily for Docker. Podman is a daemonless, rootless-capable container engine that aims to be a drop-in replacement for Docker. While Podman can be used with Dev Containers, there are fundamental architectural differences that cause compatibility issues.

## The Two Core Problems

### Problem 1: User Namespace Mapping Differences

**Docker** runs as a daemon with root privileges. When you bind-mount a directory, the container sees the files with their original host UIDs/GIDs.

**Rootless Podman** runs entirely in user space using Linux user namespaces. By default, your host UID (e.g., 1000) maps to root (UID 0) inside the container, causing permission mismatches on mounted files.

The solution is `--userns=keep-id`, which preserves your UID mapping:
- Host UID 1000 → Container UID 1000

**The catch:** Docker doesn't recognize `--userns=keep-id` and will error if you add it to `runArgs` in your devcontainer.json.

### Problem 2: Bind Mounts from External Drives Fail with `keep-id`

When `--userns=keep-id` is specified, the OCI runtime (crun/runc) must access the bind mount source path during container setup. For paths on external or separately-mounted filesystems (e.g., `/media/...`, `/mnt/...`), this access fails with:

```
Error: error stat'ing file `/media/user/Drive/path`: Permission denied: OCI permission denied
```

This occurs because:
1. The stat operation happens *before* the user namespace is fully established
2. The runtime's access to paths on mounted filesystems is restricted during this setup phase
3. The same bind mount works fine *without* `--userns=keep-id`

## The Compatibility Matrix

| Configuration | Docker | Podman (project in ~/) | Podman (project in /media) |
|---------------|--------|------------------------|---------------------------|
| No userns flag | Works | Broken UID mapping | Broken UID mapping |
| `--userns=keep-id` | Errors | Works | Permission denied |
| `PODMAN_USERNS=keep-id` | Ignored (works) | Works | Permission denied |

## Affected Scenarios

You will encounter these issues if:
- You use rootless Podman on Linux (the default installation)
- Your project resides on a secondary drive, external drive, or any mount point outside your home directory
- You want a devcontainer.json that works for teammates using Docker

## Current Workarounds

### Option 1: Move Project to Home Directory (Recommended)

The simplest solution is to keep projects that use Dev Containers within your home directory:

```bash
# Move project to home directory
mv /media/user/Data/project ~/projects/project
```

**Pros:** Works immediately, no configuration changes needed
**Cons:** May not be feasible if you need projects on a specific drive for storage reasons

### Option 2: Use Podman Machine (VM-based)

Podman Machine creates a Linux VM, similar to how Docker Desktop works on Mac/Windows:

```bash
podman machine init
podman machine start
```

Then configure VS Code to use the machine's Podman socket.

**Pros:** Avoids rootless permission issues entirely, closer to Docker Desktop behavior
**Cons:** Adds VM overhead, more complex setup

### Option 3: Symbolic Link (Partial Workaround)

Create the project in your home directory but symlink to external storage:

```bash
mkdir -p ~/projects/myproject
ln -s /media/user/Data/myproject-data ~/projects/myproject/data
```

**Pros:** Keeps large files on external storage
**Cons:** Only works for subdirectories, not the project root

### Option 4: Use Docker Instead

If your system supports it, using Docker (or Docker Desktop) avoids these issues entirely.

## For Repository Maintainers

If you maintain a repository with a devcontainer.json and want to support both Docker and Podman users:

### Do NOT add `--userns=keep-id` to runArgs

```json
// BAD - breaks Docker users
{
  "runArgs": ["--userns=keep-id"]
}
```

### Set updateRemoteUserUID appropriately

```json
{
  "updateRemoteUserUID": false,
  "remoteUser": "vscode",
  "containerUser": "vscode"
}
```

Setting `updateRemoteUserUID: false` prevents VS Code from attempting Docker-specific UID remapping that can conflict with Podman's user namespace handling.

### Document the limitation

Let Podman users know they should keep the project in their home directory, or provide alternative instructions for their setup.

## Upstream Issues

These are known issues being tracked upstream:

- **Podman:** [github.com/containers/podman/issues/18691](https://github.com/containers/podman/issues/18691) - VSCode Dev Containers fail with rootless Podman
- **VS Code:** [github.com/microsoft/vscode-remote-release/issues/10399](https://github.com/microsoft/vscode-remote-release/issues/10399) - Request for portable Docker/Podman configuration via `PODMAN_USERNS` environment variable

## Technical Details

### Why `--userns=keep-id` Causes Permission Denied on External Mounts

The sequence of operations when starting a rootless Podman container with `--userns=keep-id`:

1. Podman requests container creation from the OCI runtime (crun/runc)
2. The runtime attempts to stat the bind mount source path
3. This stat happens in an intermediate namespace state
4. For paths on external filesystems, the kernel denies access during this phase
5. The container fails before it even starts

Without `--userns=keep-id`, the stat succeeds because the namespace mapping is different, but then file ownership inside the container is wrong.

### Why Home Directory Works

Paths under your home directory (`/home/user/...`) are part of your user's primary filesystem and are accessible during all phases of user namespace setup. External mounts have different permission semantics that conflict with the namespace transition.

## Future Outlook

A complete fix would require changes to either:
- The OCI runtime's handling of bind mounts during namespace setup
- Podman's approach to setting up user namespaces with bind mounts
- VS Code Dev Containers adding native Podman support with appropriate environment variables

Until then, keeping Dev Container projects in your home directory is the most reliable approach for rootless Podman users.
