---
title: Installing Plugins
description: How to install, manage, and remove CyberEther plugins.
order: 250
category: Usage
---

Plugins expand CyberEther with custom native blocks, optional example flowgraphs, and the data files those blocks need. Each plugin is distributed as a single `.cep` bundle file. Once installed, its blocks appear right alongside built-in blocks in the catalog and behave identically.

This guide covers installing and managing plugins as a user. If you want to build and package your own plugin, read [Creating Plugins](/docs/plugins).

> [!WARNING]
> Plugins execute native code inside CyberEther with the exact same permissions as your user account. There is no sandbox. A malicious plugin can read your files, access the network, or run arbitrary code on your machine. Only install plugins you built yourself or obtained from a developer you trust.

## Security Checklist Before Installing

Take a moment to verify three things before installing any `.cep` file:

1. **Know the author.** Favor plugins from identifiable developers with public source code you can inspect.
2. **Use the official download.** Grab the `.cep` bundle directly from the developer's website or official release page rather than a third-party mirror.
3. **Verify checksums.** If the author publishes a checksum, check that your downloaded file matches.

If you cannot confirm these points, skip the plugin. During installation, CyberEther asks you to confirm that you trust the source, and that decision rests with you.

## Plugins Match Your CyberEther Version

Every plugin is built for one exact CyberEther version and one CyberEther build, such as `macos-arm64` or `linux-x86_64-cuda`. CyberEther only installs and loads a plugin that was built for the version and build you are running. After you update CyberEther, install a plugin file built for the new version.

Each `.cep` file holds one build, and its name usually says which, such as `blueprint-0.1.0-cyberether-2.0.0-linux-x86_64.cep`.

## Installing From the Gallery

The CyberEther gallery lists plugins built for every CyberEther release. To install one:

1. Open the **Start** tab and select **Plugins** next to the search field. You can also click **Browse** on the **Plugins** card in **Settings** > **Registry**.
2. Click a plugin card to open its details in a panel on the right. The panel shows its description, developer, version, license, and the blocks it provides.
3. Click **Install**.

CyberEther downloads the plugin, checks it against the checksum published by the gallery, and loads it right away.

Every time CyberEther starts, it asks the gallery for newer builds of the plugins you installed from it. Under **Settings** > **Registry** > **Updates**, the **Update Policy** setting controls what happens next:

| Policy | Behavior |
| --- | --- |
| **Ask Every Time** | The default. CyberEther lists the new versions and asks before installing them. |
| **Update Automatically** | CyberEther installs new versions in the background. |
| **Update Manually** | CyberEther only marks plugins that have a new version. |

With any policy, a plugin with a new version shows **Update** next to it in the **Plugins** card and on its detail page. New versions of plugins that are already loaded take effect the next time CyberEther starts.

After you update CyberEther itself, a gallery plugin may show **Waiting for a compatible build** until its author publishes a build for the new release. Once that build appears, CyberEther offers it like any other update.

## Installing From a File

Install a `.cep` file you downloaded or built through the Settings UI:

1. Launch CyberEther.
2. Open **CyberEther** > **Settings** from the titlebar (or press **Cmd/Ctrl+Comma**).
3. Click **Registry** in the settings sidebar.
4. Locate the **Plugins** card and click **Add**.
5. Click **Browse** to select your `.cep` file, or paste its full path into **Plugin Path**.
6. Read the trust warning and enable **Trust Source** if you trust the plugin. The install button stays disabled until both a valid path and trust confirmation are provided.
7. Click **Install Plugin**.

CyberEther copies the plugin into its own storage, loads it immediately, and opens its detail page. From then on it loads automatically every time CyberEther starts. The original `.cep` file is no longer needed, so you can move or delete it.

## Updating a Plugin

Gallery plugins follow the update policy described in [Installing From the Gallery](#installing-from-the-gallery). To update a plugin installed from a file, install the new `.cep` file the same way. Installing a file also stops gallery updates for that plugin until you install it from the gallery again.

If the plugin is already loaded, CyberEther keeps the running version and switches to the new one the next time it starts. The plugin detail page shows **Version x ready on restart** in the meantime.

## Loading Plugins Temporarily From the CLI

Pass `--plugin` on the command line to load one or more bundles for a single session without installing them:

```bash
cyberether --plugin path/to/first.cep --plugin path/to/second.cep
```

This flag also works with the `benchmark` command, including benchmarks defined inside the plugin:

```bash
cyberether benchmark blueprint_gain --plugin path/to/plugin.cep
```

A plugin loaded with `--plugin` takes priority over an installed copy of the same plugin for that run. If a `--plugin` bundle fails to load, the command exits immediately with an error.

## Using and Managing Installed Plugins

Once a plugin is loaded, its blocks show up in the normal block catalog, and any bundled example flowgraphs appear in their own section of the Start tab.

To inspect or manage a plugin, go to **Settings** > **Registry** and click an entry in the **Plugins** card. The detail page shows the plugin's status, metadata, blocks, and examples, plus one action:

- **Remove** uninstalls an installed plugin and deletes its files. Any blocks already loaded stay active until you restart CyberEther.
- **Reload** appears instead for plugins loaded with `--plugin`. It reads the `.cep` file again after you rebuild it. Close all open flowgraphs first, or CyberEther will prompt you to do so.

You can also reload every plugin loaded with `--plugin` at once from **CyberEther** > **Developer** > **Reload Plugins**.

## Flowgraphs That Use Plugins

When you save a flowgraph that uses blocks from a plugin, CyberEther records the plugin ID in the file. If you open that flowgraph on a machine without the plugin, CyberEther names the plugins it needs and offers to install them from the gallery. After they install, the flowgraph opens.

## Status Reference and Troubleshooting

Check the **Status** field on a plugin's detail page to diagnose load issues:

| Status | Meaning |
|---|---|
| **Loaded** | The plugin is active and its blocks are ready to use. |
| **Loaded for this run** | The plugin was loaded with `--plugin` and is not installed. |
| **Ready on restart** | The plugin is installed and loads the next time CyberEther starts. |
| **Replacement file needed** | The installed plugin was built for another CyberEther version or build. Install a matching `.cep` file. |
| **Waiting for a compatible build** | The gallery has no build of this plugin for your CyberEther version yet. Once it appears, CyberEther offers it as an update. |
| **Failed to load** | The plugin is installed but did not load. Check the logs for the exact error. |

If an install shows an error, the cause is usually one of these:

- **Different CyberEther version.** The plugin was built for another CyberEther release. Ask the developer for a build that matches your version.
- **Gallery unreachable.** CyberEther could not contact the gallery or the download did not match its checksum. Check your connection and try again.
- **Different build.** The file was built for another platform, for example a macOS Apple Silicon file on Linux, or a CUDA file on a build without CUDA. Download the file that matches your build.

To see the exact error message, launch CyberEther from a terminal and look for log lines starting with `[PLUGIN]`.
