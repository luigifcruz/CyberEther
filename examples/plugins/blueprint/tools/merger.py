#!/usr/bin/env python3

import argparse
import io
import tarfile
import tempfile
from pathlib import Path


def yaml_quote(value):
    return "'" + value.replace("'", "''") + "'"


def parse_scalar(value):
    value = value.strip()
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    return value


def parse_manifest(text):
    metadata = {}
    targets = []
    examples = []
    section = None
    current = None

    for raw in text.splitlines():
        if not raw.strip():
            continue

        if not raw.startswith(" "):
            key, _, rest = raw.partition(":")
            section = key.strip()
            current = None
            if rest.strip() == "[]":
                section = None
            continue

        line = raw.strip()
        if section == "metadata":
            key, _, value = line.partition(":")
            metadata[key.strip()] = parse_scalar(value)
        elif section in ("targets", "examples"):
            if line.startswith("- "):
                current = {}
                (targets if section == "targets" else examples).append(current)
                line = line[2:]
            if current is None:
                raise SystemExit(f"malformed manifest entry: {raw}")
            key, _, value = line.partition(":")
            current[key.strip()] = parse_scalar(value)

    return metadata, targets, examples


def add_bytes(tar, arcname, data, mode=0o644):
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def load_bundle(path):
    with tarfile.open(path, "r:gz") as tar:
        member = tar.extractfile("manifest.yml")
        if member is None:
            raise SystemExit(f"bundle has no manifest.yml: {path}")
        metadata, targets, examples = parse_manifest(member.read().decode("utf-8"))

        for entry in targets + examples:
            file = tar.extractfile(entry["path"])
            if file is None:
                raise SystemExit(f"bundle member missing: {entry['path']} in {path}")
            entry["data"] = file.read()
            entry["mode"] = tar.getmember(entry["path"]).mode

    return metadata, targets, examples


def build_manifest(metadata, targets, examples):
    lines = ["metadata:"]
    for key in ("name", "version", "minimumJetstreamVersion"):
        lines.append(f"  {key}: {yaml_quote(metadata[key])}")

    lines += ["", "targets:"]
    for target in targets:
        lines += [
            f"  - path: {yaml_quote(target['path'])}",
            f"    system: {yaml_quote(target['system'])}",
            f"    device: {yaml_quote(target['device'])}",
            f"    arch: {yaml_quote(target['arch'])}",
        ]

    lines += ["", "examples:"]
    if examples:
        for example in examples:
            lines.append(f"  - path: {yaml_quote(example['path'])}")
    else:
        lines[-1] = "examples: []"

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Merge single-target .cep bundles into one multi-target bundle."
    )
    parser.add_argument("--output", required=True, help="Output .cep path.")
    parser.add_argument("inputs", nargs="+", help="Input .cep bundles to merge.")
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix.lower() != ".cep":
        raise SystemExit("output path must end with .cep")

    metadata = None
    targets = []
    examples = {}
    seen_variants = set()

    for input_path in args.inputs:
        bundle_metadata, bundle_targets, bundle_examples = load_bundle(input_path)

        if metadata is None:
            metadata = bundle_metadata
        elif metadata != bundle_metadata:
            raise SystemExit(f"bundle metadata does not match: {input_path}")

        for target in bundle_targets:
            variant = (target["system"], target["device"], target["arch"])
            if variant in seen_variants:
                raise SystemExit(
                    "duplicate target " + "-".join(variant) + f": {input_path}"
                )
            seen_variants.add(variant)
            targets.append(target)

        for example in bundle_examples:
            known = examples.get(example["path"])
            if known is None:
                examples[example["path"]] = example
            elif known["data"] != example["data"]:
                raise SystemExit(
                    f"example content does not match: {example['path']} in {input_path}"
                )

    manifest = build_manifest(metadata, targets, list(examples.values()))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with tarfile.open(temp_path, "w:gz", format=tarfile.USTAR_FORMAT) as tar:
            add_bytes(tar, "manifest.yml", manifest.encode("utf-8"))
            for target in targets:
                add_bytes(tar, target["path"], target["data"], target["mode"])
            for example in examples.values():
                add_bytes(tar, example["path"], example["data"], example["mode"])

        temp_path.replace(output)
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
