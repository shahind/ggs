#!/bin/sh
# GGS starter for macOS (Apple Silicon, macOS 14+) and Linux: installs Python 3.13, the libraries and the motion
# library into .venv.
#   ./setup.sh                    everything except the AI models (the app offers them on first start)
#   ./setup.sh --models required  also download the AI models this computer needs
# Running it again repairs or updates the install. Then start the app:  .venv/bin/python -m ggs.ui.app
set -e
cd "$(dirname "$0")"
UV_VERSION=0.13.9
if command -v uv >/dev/null 2>&1; then
  UV=uv
else
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)  ASSET=uv-aarch64-apple-darwin.tar.gz
                   SHA=5aeca67f2283f75235de40d0bd9c0a5bbc5fce23b6d48b3bf9fa1b2db08382ca ;;
    Linux-x86_64)  ASSET=uv-x86_64-unknown-linux-gnu.tar.gz
                   SHA=a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5 ;;
    Linux-aarch64) ASSET=uv-aarch64-unknown-linux-gnu.tar.gz
                   SHA=0797542018e541a9288855df555c8077eed06f7a47d206362030480e8ecc9f3d ;;
    Darwin-*)      echo "GGS needs a Mac with Apple Silicon (M1 or newer); PyTorch no longer supports Intel Macs."
                   exit 1 ;;
    *)             echo "Unsupported system: $(uname -s) $(uname -m)"; exit 1 ;;
  esac
  DIR="$HOME/.ggs/tools/uv-$UV_VERSION"
  UV="$DIR/uv"
  if [ ! -x "$UV" ]; then
    echo "Downloading uv $UV_VERSION (Python installer, about 20 MB)..."
    mkdir -p "$DIR"
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$ASSET" -o "$DIR/$ASSET"
    GOT=$( (shasum -a 256 "$DIR/$ASSET" 2>/dev/null || sha256sum "$DIR/$ASSET") | cut -d' ' -f1)
    if [ "$GOT" != "$SHA" ]; then rm -f "$DIR/$ASSET"; echo "uv download is damaged ($GOT)"; exit 1; fi
    tar -xzf "$DIR/$ASSET" -C "$DIR" --strip-components 1
    rm -f "$DIR/$ASSET"
  fi
fi
exec "$UV" run --no-project --python 3.13 tools/setup_env.py "$@"
