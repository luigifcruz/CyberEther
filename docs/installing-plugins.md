---
title: Installing Plugins
description: How to install, manage, and remove CyberEther plugins.
order: 250
category: Usage
---

Plugins expand CyberEther with custom native blocks and optional example flowgraphs. Each plugin is distributed as a single `.cep` bundle file. Once loaded, its blocks appear right alongside built-in blocks in the catalog and behave identically.

This guide covers installing and managing plugins as a user. If you want to build and package your own plugin, read [Creating Plugins](/docs/plugins).

> [!WARNING]
> Plugins execute native code inside CyberEther with the exact same permissions as your user account. There is no sandbox. A malicious plugin can read your files, access the network, or run arbitrary code on your machine. Only install plugins you built yourself or obtained from a developer you trust.

> [!NOTE]
> Plugin bundles are experimental. The bundle format and loader ABI can change between CyberEther releases, so plugins may occasionally need to be rebuilt for newer versions.

## Security Checklist Before Installing

Take a moment to verify three things before registering any `.cep` file:

1. **Know the author.** Favor plugins from identifiable developers with public source code you can inspect.
2. **Use the official download.** Grab the `.cep` bundle directly from the developer's website or official release page rather than a third-party mirror.
3. **Verify checksums.** If the author publishes a checksum, check that your downloaded file matches.

If you cannot confirm these points, skip the plugin. During installation, CyberEther asks you to confirm that you trust the source, and that decision rests with you.

## Installing a Plugin Permanently

First, move your `.cep` file to a permanent folder on disk, such as a `Plugins` directory in your home folder. CyberEther loads plugins from their registered path on every startup rather than copying the file into internal storage.

Then register it through the Settings UI:

1. Launch CyberEther.
2. Open **CyberEther** > **Settings** from the titlebar (or press **Cmd/Ctrl + Comma**).
3. Click **Registry** in the settings sidebar.
4. Locate the **Plugins** card and click **Add**.
5. Click **Browse** to select your `.cep` file, or paste its full path into **Plugin Path**.
6. Read the trust warning and enable **Trust Source** if you trust the plugin. The register button remains disabled until both a valid path and trust confirmation are provided.
7. Click **Register Plugin**.

CyberEther loads the bundle immediately, displays a "Plugin loaded." notification, and adds it to the Plugins table with its name, version, and status. Going forward, it loads automatically every time CyberEther starts. If a registered plugin fails to load at startup, CyberEther logs a warning and continues launching normally.

> [!IMPORTANT]
> **Keep the `.cep` file in place.** Because CyberEther remembers the file path instead of copying the bundle, moving, renaming, or deleting the file causes the plugin to show a **Missing** status on the next launch. If you relocate a bundle, remove the stale entry and register it again from its new path.

## Loading Plugins Temporarily From the CLI

Pass `--plugin` on the command line to load one or more bundles for a single session without modifying `settings.yaml`:

```bash
cyberether --plugin path/to/first.cep --plugin path/to/second.cep
```

This flag also works with the `benchmark` command, including benchmarks defined inside the plugin:

```bash
cyberether benchmark custom_block --plugin path/to/plugin.cep
```

Command-line plugins load alongside any registered plugins for that run only. Unlike startup plugins, if an explicitly passed `--plugin` bundle fails to load, the command exits immediately with an error.

## Using and Managing Installed Plugins

Once a plugin is loaded, its blocks show up in the normal block catalog, and any bundled example flowgraphs appear in their own section on the **Start** tab so you can open them directly.

To inspect or manage a plugin, go to **Settings** > **Registry** and click an entry in the **Plugins** card. The detail page shows the plugin's status, metadata, blocks, examples, and two actions:

- **Reload** unloads and reloads the bundle after you replace the `.cep` file with an updated build. Close all open flowgraphs first, or CyberEther will prompt you to do so.
- **Remove** unregisters the plugin from your list. Any blocks already loaded stay active until you restart CyberEther, and the `.cep` file on disk is left untouched.

You can also reload every registered plugin at once from **CyberEther** > **Developer** > **Reload Plugins**.

## Status Reference and Troubleshooting

Check the **Status** field on a plugin's detail page to diagnose load issues:

| Status | Meaning |
|---|---|
| **Loaded** | The plugin is active and its blocks are ready to use. |
| **Not loaded** | The plugin is registered but failed to load. Check the logs for the exact error. |
| **Missing** | No file exists at the registered path. |
| **Invalid extension** | The registered path does not end in `.cep`. |

If registration triggers a "Failed to load plugin." notification, the cause is usually one of three things:

- **Outdated CyberEther version.** Every `.cep` bundle specifies a minimum supported version. Update CyberEther and retry.
- **No matching target platform.** Bundles contain compiled libraries for specific operating systems, architectures, and device backends (for example, a macOS Apple Silicon bundle cannot load on Linux). Request a build matching your platform from the developer.
- **Plugin ABI mismatch.** Because plugin bundles are experimental, a plugin compiled for an older CyberEther release may be rejected by a newer one. Ask the developer for an updated build.

To see the exact error message, launch CyberEther from a terminal and look for log lines starting with `[PLUGIN]`.

### Platform Notes

- **Browser edition:** WebAssembly side modules cannot be unloaded in place, so refresh the browser tab after replacing a plugin.
- **Temporary CLI plugins:** Bundles loaded with `--plugin` are not persisted. Register them through **Settings** if you want them loaded automatically on every launch.
