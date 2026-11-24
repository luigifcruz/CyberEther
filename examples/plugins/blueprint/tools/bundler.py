#!/usr/bin/env python3

import argparse
import json
import re
import shutil
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


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
RESERVED_NAMES = RESERVED_IDS | {f"{device}{digit}" for device in ("com", "lpt") for digit in "\u00b9\u00b2\u00b3"}
MAX_BUNDLE_SIZE = 512 << 20
MAX_UNCOMPRESSED_SIZE = 2 << 30
MAX_ENTRIES = 10000
MAX_MANIFEST_SIZE = 64 << 10


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


def zip_info(arcname, mode):
    info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100000 | mode) << 16
    return info


def add_bytes(archive, arcname, data, mode=0o644):
    archive.writestr(zip_info(arcname, mode), data)


def add_file(archive, source, arcname):
    if not source.is_file():
        raise SystemExit(f"input file does not exist: {source}")

    mode = source.stat().st_mode & 0o777
    with source.open("rb") as file, archive.open(zip_info(arcname, mode), "w") as target:
        shutil.copyfileobj(file, target)


def check_path(archive_path):
    for segment in archive_path.split("/"):
        if (
            not segment
            or segment[-1] in ". "
            or any(character in segment for character in "\\:\0")
            or segment.split(".", 1)[0].strip().lower() in RESERVED_NAMES
        ):
            raise SystemExit(f"bundle path is not portable: {archive_path}")


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
        check_path(archive_path)
        if archive_path.lower() in archive_paths:
            raise SystemExit(f"duplicate bundle path: {archive_path}")
        archive_paths.add(archive_path.lower())
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
    if len(manifest) > MAX_MANIFEST_SIZE:
        raise SystemExit("manifest.yml must be at most 64 KiB")
    if len(entries) + 1 > MAX_ENTRIES:
        raise SystemExit(f"bundles can hold at most {MAX_ENTRIES} entries")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED, allowZip64=False) as archive:
            add_bytes(archive, "manifest.yml", manifest)
            for source, archive_path in entries:
                add_file(archive, source, archive_path)
            if sum(info.file_size for info in archive.infolist()) > MAX_UNCOMPRESSED_SIZE:
                raise SystemExit("bundle contents must be at most 2 GiB")

        if temp_path.stat().st_size > MAX_BUNDLE_SIZE:
            raise SystemExit("bundle must be at most 512 MiB")
        temp_path.replace(output)
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
