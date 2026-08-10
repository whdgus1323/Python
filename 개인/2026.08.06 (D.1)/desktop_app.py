from __future__ import annotations

import sys
from pathlib import Path

SOURCE_DIRECTORY = Path(__file__).parent / "src"
PROJECT_SITE_PACKAGES = Path(__file__).parent / ".venv" / "Lib" / "site-packages"


def main() -> int:
    for directory in (SOURCE_DIRECTORY, PROJECT_SITE_PACKAGES):
        directory_text = str(directory)
        if directory_text not in sys.path:
            sys.path.insert(0, directory_text)
    from keyword_sourcing.desktop import run_desktop

    return run_desktop(smoke_test="--smoke-test" in sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
