#!/bin/sh
# Installs the krino CLI from the latest GitHub Release.
#
#   curl -sSf https://raw.githubusercontent.com/getkrino/krino/main/install.sh | sh
#
# Only linux-x64 release binaries are published today. Other platforms
# must build from source: `make install` (see CONTRIBUTING.md).
set -eu

REPO="getkrino/krino"
BINARY="krino-linux-x64"

os=$(uname -s)
arch=$(uname -m)

if [ "$os" != "Linux" ] || [ "$arch" != "x86_64" ]; then
    echo "error: no prebuilt krino binary for $os/$arch." >&2
    echo "Only linux-x64 is published (see docs/cli.md#platform-support)." >&2
    echo "Build from source instead:" >&2
    echo "  git clone https://github.com/$REPO && cd krino && make install" >&2
    exit 1
fi

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
