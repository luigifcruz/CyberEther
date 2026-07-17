#!/usr/bin/env python3

import argparse
import io
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


def normalize_system(value):
    value = value.lower()
    aliases = {
        "darwin": "macos",
        "mac": "macos",
        "osx": "macos",
        "win32": "windows",
        "mingw": "windows",
    }
    return aliases.get(value, value)


def normalize_arch(value):
    value = value.lower()
    aliases = {
        "aarch64": "arm64",
        "amd64": "x86_64",
        "x64": "x86_64",
    }
    return aliases.get(value, value)


def yaml_quote(value):
    return "'" + value.replace("'", "''") + "'"


def parse_target(value):
    fields = {}
    for part in value.split(","):
        key, separator, field_value = part.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(
                "targets must use key=value fields separated by commas"
            )
        fields[key.strip()] = field_value.strip()

    required = {"path", "system", "device", "arch"}
    missing = sorted(required - set(fields))
    if missing:
        raise argparse.ArgumentTypeError(
            "target is missing required fields: " + ", ".join(missing)
        )

    return {
        "source": Path(fields["path"]),
        "system": normalize_system(fields["system"]),
        "device": fields["device"].lower(),
        "arch": normalize_arch(fields["arch"]),
    }


def add_bytes(tar, arcname, data):
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def add_file(tar, source, arcname):
    if not source.is_file():
        raise SystemExit(f"input file does not exist: {source}")

    info = tar.gettarinfo(str(source), arcname)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o755 if os.access(source, os.X_OK) else 0o644

    with source.open("rb") as file:
        tar.addfile(info, file)


def build_manifest(args, targets, examples):
    lines = [
        "metadata:",
        f"  name: {yaml_quote(args.name)}",
        f"  version: {yaml_quote(args.version)}",
        f"  minimumJetstreamVersion: {yaml_quote(args.minimum_jetstream_version)}",
        "",
        "targets:",
    ]

    for target in targets:
        lines += [
            f"  - path: {yaml_quote(target['archive_path'])}",
            f"    system: {yaml_quote(target['system'])}",
            f"    device: {yaml_quote(target['device'])}",
            f"    arch: {yaml_quote(target['arch'])}",
        ]

    lines += ["", "examples:"]
    if examples:
        for example in examples:
            lines.append(f"  - path: {yaml_quote(example['archive_path'])}")
    else:
        lines[-1] = "examples: []"

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Create a CyberEther .cep plugin bundle.")
    parser.add_argument("--output", required=True, help="Output .cep path.")
    parser.add_argument("--name", required=True, help="Plugin name.")
    parser.add_argument("--version", required=True, help="Plugin version.")
    parser.add_argument(
        "--minimum-jetstream-version",
        "--min-jetstream-version",
        dest="minimum_jetstream_version",
        required=True,
        help="Minimum CyberEther/Jetstream version, for example 1.7.0.",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        required=True,
        help="Target as path=...,system=...,device=...,arch=.... May be repeated.",
    )
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help="Example flowgraph path. May be repeated.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix.lower() != ".cep":
        raise SystemExit("output path must end with .cep")

    targets = []
    archive_paths = set()
    for target in args.target:
        archive_path = PurePosixPath(
            "targets",
            f"{target['system']}-{target['arch']}-{target['device']}",
            target["source"].name,
        ).as_posix()
        if archive_path in archive_paths:
            raise SystemExit(f"duplicate bundle path: {archive_path}")
        archive_paths.add(archive_path)
        targets.append({**target, "archive_path": archive_path})

    examples = []
    for example in args.example:
        source = Path(example)
        archive_path = PurePosixPath("examples", source.name).as_posix()
        if archive_path in archive_paths:
            raise SystemExit(f"duplicate bundle path: {archive_path}")
        archive_paths.add(archive_path)
        examples.append({"source": source, "archive_path": archive_path})

    manifest = build_manifest(args, targets, examples).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with tarfile.open(temp_path, "w:gz", format=tarfile.USTAR_FORMAT) as tar:
            add_bytes(tar, "manifest.yml", manifest)
            for target in targets:
                add_file(tar, target["source"], target["archive_path"])
            for example in examples:
                add_file(tar, example["source"], example["archive_path"])

        temp_path.replace(output)
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
