import gc
import importlib
import importlib.machinery
import importlib.metadata
import os
import sys


def _jetstream_package_names(path):
    names = set()
    if not path:
        return names

    for distribution in importlib.metadata.distributions(path=[path]):
        top_level = distribution.read_text("top_level.txt")
        if top_level:
            names.update(name for name in top_level.splitlines() if name.isidentifier())
            continue

        for file in distribution.files or ():
            part = file.parts[0]
            name = part[:-3] if part.endswith(".py") else part.partition(".")[0]
            if name.isidentifier():
                names.add(name)
    return names


def _jetstream_inside(path, root):
    if not path or not root:
        return False
    try:
        return os.path.commonpath((os.path.realpath(path), root)) == root
    except (OSError, ValueError):
        return False


def _jetstream_native_packages():
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    names = set()
    for name, module in tuple(sys.modules.items()):
        loader = getattr(getattr(module, "__spec__", None), "loader", None)
        path = getattr(module, "__file__", None) or ""
        if isinstance(loader, importlib.machinery.ExtensionFileLoader) or path.endswith(suffixes):
            names.add(name.partition(".")[0])
    return names


def _jetstream_switch_packages(previous, current):
    previous = os.path.realpath(previous) if previous else ""
    current = os.path.realpath(current) if current else ""
    if previous == current:
        return []

    supplied = _jetstream_package_names(current)
    native = _jetstream_native_packages()
    for name, module in tuple(sys.modules.items()):
        if name.partition(".")[0] in native:
            continue
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin in ("built-in", "frozen"):
            continue
        path = getattr(module, "__file__", None)
        if _jetstream_inside(path, previous) or name.partition(".")[0] in supplied:
            sys.modules.pop(name, None)

    sys.path[:] = [
        path
        for path in sys.path
        if not (
            (previous and os.path.realpath(path) == previous)
            or (current and os.path.realpath(path) == current)
        )
    ]
    if current:
        sys.path.insert(0, current)

    if previous:
        sys.path_importer_cache.pop(previous, None)
    if current:
        sys.path_importer_cache.pop(current, None)
    importlib.invalidate_caches()
    gc.collect()

    kept = []
    for name in sorted(native & supplied):
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None) or next(iter(getattr(module, "__path__", ())), None)
        if module is not None and not _jetstream_inside(path, current):
            kept.append(name)
    return kept
