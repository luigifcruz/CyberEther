import json
import os
import platform
import site
import sys

def user_site_enabled():
    if os.environ.get("PYTHONNOUSERSITE"):
        return False
    if hasattr(os, "getuid") and hasattr(os, "geteuid"):
        if os.getuid() != os.geteuid():
            return False
    if hasattr(os, "getgid") and hasattr(os, "getegid"):
        if os.getgid() != os.getegid():
            return False

    executable_directory = os.path.dirname(os.path.abspath(sys.executable))
    prefix = os.path.dirname(executable_directory)
    for config in (
        os.path.join(executable_directory, "pyvenv.cfg"),
        os.path.join(prefix, "pyvenv.cfg"),
    ):
        if not os.path.isfile(config):
            continue
        include_system_site_packages = "true"
        with open(config, encoding="utf-8") as file:
            for line in file:
                key, separator, value = line.partition("=")
                if separator and key.strip().lower() == "include-system-site-packages":
                    include_system_site_packages = value.strip().lower()
        return include_system_site_packages == "true"

    return True


def add_user_site():
    if not user_site_enabled():
        return

    paths = site.getusersitepackages()
    if isinstance(paths, str):
        paths = [paths]

    existing_paths = set(sys.path)
    for path in paths:
        site.addsitedir(path)

    added_paths = [path for path in sys.path if path not in existing_paths]
    if added_paths:
        sys.path[:] = [path for path in sys.path if path not in added_paths]
        system_sites = {
            os.path.realpath(path)
            for path in site.getsitepackages(
                [sys.base_prefix, sys.base_exec_prefix]
            )
        }
        insertion = next(
            (
                index
                for index, path in enumerate(sys.path)
                if os.path.realpath(path) in system_sites
            ),
            len(sys.path),
        )
        sys.path[insertion:insertion] = added_paths


def emit(valid, error=""):
    print(
        "JETSTREAM_PEP723_RESULT="
        + json.dumps(
            {
                "valid": valid,
                "error": error,
            },
            ensure_ascii=True,
        )
    )


try:
    add_user_site()
    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
            from pip._vendor.packaging.specifiers import SpecifierSet
            from pip._vendor.packaging.version import Version
        except ImportError:
            emit(
                False,
                "The selected Python runtime provides neither packaging nor pip's "
                "vendored packaging module.",
            )
            raise SystemExit

    requires_python = sys.argv[1]
    for raw in sys.argv[2:]:
        try:
            Requirement(raw)
        except Exception as error:
            emit(False, f"Invalid PEP 508 requirement {raw!r}: {error}")
            raise SystemExit

    try:
        specifiers = SpecifierSet(requires_python)
    except Exception as error:
        emit(False, f"Invalid PEP 723 requires-python {requires_python!r}: {error}")
        raise SystemExit

    version = Version(platform.python_version())
    if requires_python and not specifiers.contains(version, prereleases=True):
        emit(
            False,
            f"Selected Python {version} does not satisfy PEP 723 requires-python "
            f"{requires_python!r}.",
        )
        raise SystemExit

    emit(True)
except SystemExit:
    pass
except Exception as error:
    emit(False, f"Python requirement preflight failed: {error}")
