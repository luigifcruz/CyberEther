#!/usr/bin/env python3

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def patch_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    content = content.replace("fmt::", "jst::fmt::")
    content = content.replace("namespace fmt", "namespace jst::fmt")
    content = content.replace("FMT_", "JST_FMT_")
    path.write_text(content, encoding="utf-8")


def main() -> int:
    fmt_path = Path(sys.argv[1]).resolve()
    fmt_root = fmt_path.parent
    stamp_path = fmt_root / "patch_headers.stamp"
    jetstream_fmt_root = fmt_root / "jetstream" / "fmt"

    if not stamp_path.exists():
        for source in fmt_path.rglob("*"):
            if source.suffix in {".h", ".cc"}:
                patch_file(source)

    if jetstream_fmt_root.exists():
        shutil.rmtree(jetstream_fmt_root)

    jetstream_fmt_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fmt_path, jetstream_fmt_root)
    stamp_path.touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
