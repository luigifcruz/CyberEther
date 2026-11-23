#!/usr/bin/env python3

import argparse
import io
import json
import re
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


ID_PATTERN = re.compile(r"^[a-z][a-z0-9]{0,31}$")
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}$")
BUILD_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9_]+)+$")
KNOWN_BUILDS = {
    "macos-arm64",
    "linux-x86_64",
    "linux-x86_64-cuda",
    "linux-aarch64",
    "linux-aarch64-cuda",
    "windows-x86_64",
}


RESERVED_IDS = {"aux", "con", "nul", "prn"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}


def parse_id(value):
    if not ID_PATTERN.match(value) or value in RESERVED_IDS:
        raise argparse.ArgumentTypeError(
            "id must use 1 to 32 lowercase letters and digits, starting with a letter, "
            "and avoid names reserved by Windows"
        )
    return value


def parse_version(value):
    if not VERSION_PATTERN.match(value) or ".." in value:
        raise argparse.ArgumentTypeError(
            "version must use letters, digits, '.', '-' or '+', starting with a letter or digit"
        )
    return value


def parse_build(value):
    if not BUILD_PATTERN.match(value):
        raise argparse.ArgumentTypeError("build must be a CyberEther build ID such as linux-x86_64")
    return value


def library_extension(build_id):
    if build_id.startswith("macos-"):
        return ".dylib"
    if build_id.startswith("windows-"):
        return ".dll"
    return ".so"


def tar_info(arcname, size, mode):
    info = tarfile.TarInfo(arcname)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def add_bytes(tar, arcname, data, mode=0o644):
    tar.addfile(tar_info(arcname, len(data), mode), io.BytesIO(data))


def add_file(tar, source, arcname):
    if not source.is_file():
        raise SystemExit(f"input file does not exist: {source}")

    mode = source.stat().st_mode & 0o777
    with source.open("rb") as file:
        tar.addfile(tar_info(arcname, source.stat().st_size, mode), file)


def collect_files(source, prefix):
    if source.is_file():
        return [(source, PurePosixPath(prefix, source.name).as_posix())]
    if not source.is_dir():
        raise SystemExit(f"input path does not exist: {source}")

    files = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise SystemExit(f"symbolic links are not supported, store the file itself: {path}")
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            files.append((path, PurePosixPath(prefix, relative).as_posix()))
    return files


def build_manifest(args):
    fields = [
        ("id", args.id),
        ("title", args.title),
        ("version", args.version),
        ("cyberether", args.cyberether),
        ("build", args.build),
        ("summary", args.summary),
        ("author", args.author),
        ("license", args.license),
        ("description", args.description),
    ]
    lines = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in fields if value]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Create a CyberEther .cep plugin bundle.")
    parser.add_argument("--output", required=True, help="Output .cep path.")
    parser.add_argument("--id", required=True, type=parse_id, help="Stable plugin ID.")
    parser.add_argument("--title", required=True, help="Display name.")
    parser.add_argument("--version", required=True, type=parse_version, help="Plugin version.")
    parser.add_argument(
        "--cyberether",
        required=True,
        type=parse_version,
        help="Exact CyberEther version the plugin was built against.",
    )
    parser.add_argument("--summary", default="", help="One line shown in plugin lists.")
    parser.add_argument("--description", default="", help="Longer Markdown description.")
    parser.add_argument("--author", default="", help="Plugin author.")
    parser.add_argument("--license", default="", help="Plugin license.")
    parser.add_argument(
        "--build",
        required=True,
        type=parse_build,
        help="CyberEther build the library was compiled for, such as linux-x86_64.",
    )
    parser.add_argument("--library", required=True, help="Plugin library. It is stored as lib/<id>.<ext>.")
    parser.add_argument(
        "--lib-file",
        action="append",
        default=[],
        help="Extra file or folder copied into lib/, such as a bundled library. May be repeated.",
    )
    parser.add_argument(
        "--data",
        action="append",
        default=[],
        help="File or folder copied into data/ and shared by every build. May be repeated.",
    )
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help="Example flowgraph (.yml). May be repeated.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix.lower() != ".cep":
        raise SystemExit("output path must end with .cep")

    entries = []
    archive_paths = set()

    def claim(source, archive_path):
        if archive_path in archive_paths:
            raise SystemExit(f"duplicate bundle path: {archive_path}")
        archive_paths.add(archive_path)
        entries.append((source, archive_path))

    if args.build not in KNOWN_BUILDS:
        print(f"warning: '{args.build}' is not a known CyberEther release build")
    claim(Path(args.library), PurePosixPath("lib", args.id + library_extension(args.build)).as_posix())

    for source in args.lib_file:
        for path, archive_path in collect_files(Path(source), "lib"):
            claim(path, archive_path)

    for data in args.data:
        for path, archive_path in collect_files(Path(data), "data"):
            claim(path, archive_path)

    for example in args.example:
        source = Path(example)
        if source.suffix != ".yml":
            raise SystemExit(f"examples must be .yml files: {source}")
        claim(source, PurePosixPath("examples", source.name).as_posix())

    manifest = build_manifest(args).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with tarfile.open(temp_path, "w:gz", format=tarfile.USTAR_FORMAT) as tar:
            add_bytes(tar, "manifest.yml", manifest)
            for source, archive_path in entries:
                add_file(tar, source, archive_path)

        temp_path.replace(output)
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
