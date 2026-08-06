#!/bin/sh
# Installs the krino CLI from the latest GitHub Release.
#
#   curl -sSf https://raw.githubusercontent.com/getkrino/krino/main/install.sh | sh
#
# Supports linux-x64, macos-x64, and macos-arm64. Windows binaries are
# published too (krino-windows-x64.exe) but this is a POSIX script —
# Windows users should download the .exe from GitHub Releases directly,
# or build from source (see docs/cli.md#platform-support).
set -eu

REPO="getkrino/krino"

os=$(uname -s)
arch=$(uname -m)

case "$os" in
    Linux)
        case "$arch" in
            x86_64) target="linux-x64" ;;
            *) target="" ;;
        esac
        ;;
    Darwin)
        case "$arch" in
            x86_64) target="macos-x64" ;;
            arm64) target="macos-arm64" ;;
            *) target="" ;;
        esac
        ;;
    *)
        target=""
        ;;
esac

if [ -z "$target" ]; then
    echo "error: no prebuilt krino binary for $os/$arch." >&2
    echo "Supported: linux-x64, macos-x64, macos-arm64 (see docs/cli.md#platform-support)." >&2
    echo "Windows: download krino-windows-x64.exe from GitHub Releases directly." >&2
    echo "Otherwise, build from source:" >&2
    echo "  git clone https://github.com/$REPO && cd krino && make install" >&2
    exit 1
fi

BINARY="krino-$target"

if [ -w /usr/local/bin ]; then
    install_dir="/usr/local/bin"
else
    install_dir="$HOME/.local/bin"
    mkdir -p "$install_dir"
fi

url="https://github.com/$REPO/releases/latest/download/$BINARY"
dest="$install_dir/krino"

echo "Downloading $url"
if command -v curl >/dev/null 2>&1; then
    curl -sSfL "$url" -o "$dest"
elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$dest"
else
    echo "error: need curl or wget to install krino." >&2
    exit 1
fi

chmod +x "$dest"
echo "Installed krino to $dest"

case ":$PATH:" in
    *":$install_dir:"*) ;;
    *)
        echo
        echo "warning: $install_dir is not on your PATH. Add it, e.g.:"
        echo "  export PATH=\"$install_dir:\$PATH\""
        ;;
esac

echo
"$dest" version
