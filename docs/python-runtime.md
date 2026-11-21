---
title: Python Runtime And Dependencies
description: Configure the Python interpreter, declare PEP 723 dependencies, and manage environment caching and install policies.
order: 231
category: Usage
parent: python-block
---

CyberEther embeds Python directly in process so your [Python Block](/docs/python-block) scripts can access tensors with zero copies and pull in third-party libraries cleanly.

## Choosing a Python Runtime

On native desktop and server platforms, CyberEther does not bundle its own Python interpreter. Instead, it embeds a Python installation already present on your system, giving your scripts access to the same packages you use from your terminal. Scripts can also declare extra packages through [inline dependency metadata](#declaring-dependencies).

Open **Settings** > **Runtime** and use the **Python Runtime** selector:

- **Auto (default)** scans your `PATH` along with active `venv`, Conda, and pyenv environments, picking the first usable interpreter. Launching CyberEther from a terminal with your environment activated is enough for Auto to find it.
- **Detected installations** appear in the dropdown with their version and path, such as `Python 3.12.4 (/opt/miniconda3/bin/python)`.
- **Custom Path** lets you enter or browse to any `python` executable (paths starting with `~` are supported).

A status badge under **Interpreter** shows whether a matching Python shared library was found or reports why the selected interpreter is unusable. Runtime changes are saved and take effect after restarting CyberEther. Python 3.9 is the minimum supported version, and older installations are marked invalid and skipped by Auto.

> [!TIP]
> If `python -c "import numpy"` succeeds in your terminal, selecting that exact interpreter in Settings makes `import numpy` work inside the Python block too. When an import fails, verify which runtime is selected before reinstalling packages.

> [!NOTE]
> In the [Web Edition](/docs/web), CyberEther runs a browser-compiled CPython environment inside WebAssembly rather than selecting a local system interpreter. System `pip` installs and CUDA/CuPy packages are only available in native builds.

## Declaring Dependencies

Python blocks support PEP 723 inline script metadata. List the packages your script imports directly at the top of the file:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0,<3",
#   "scipy~=1.14",
#   "requests>=2.32,<3",
# ]
# ///

def compute(ctx):
    pass
```

Only declare direct imports because transitive dependencies are resolved automatically. Prefer broad version ranges that match what you have tested, and reserve exact pins for hard requirements.

## How Environments Are Resolved and Cached

Because every Python block in a CyberEther process shares one interpreter, CyberEther resolves the union of dependencies declared across all scheduled blocks. Scripts with no metadata header can still import packages contributed by other blocks. If two blocks declare incompatible constraints, the new environment refuses to activate and reports the conflict on the requesting blocks.

Resolved environments are cached under `python-environments/<environment-key>/` inside CyberEther's cache directory, keyed by the active Python runtime and dependency union. Each entry is immutable and stores its direct requirements, a `site-packages` directory, and a completion marker (incomplete entries are ignored). CyberEther runs `pip` from the chosen runtime, leaving package resolution and download/wheel caching to `pip`.

## Dependency Policy

Under **Settings** > **Runtime** > **Dependencies**, the **Installation Policy** setting controls what happens when a required environment is not yet cached:

| Settings UI Label | CLI Value (`--dependency-policy`) | Behavior |
|---|---|---|
| **Ask Every Time** | `prompt` | Asks for your approval before installing a missing environment. |
| **Install Automatically** | `allow` | Installs missing environments automatically. |
| **Never Install** | `deny` | Uses already-cached environments and never installs packages. |

In headless mode with **Ask Every Time** (`prompt`), blocks waiting on missing packages pause until someone approves the request through a connected interface such as [Remote Streaming](/docs/remote). For fully unattended deployments, select **Never Install** (`deny`) or pre-populate the cache ahead of time. Only pass `--dependency-policy allow` when you trust the flowgraph scripts and want automatic installation.
