---
title: Installing Plugins
description: How to install, manage, and remove CyberEther plugins.
order: 42
category: Usage
---

> [!WARNING]
>
> Plugins run native code inside CyberEther with the same permissions as CyberEther itself. There is no sandbox. Installing a plugin is like installing any vendored app on your computer. A malicious plugin can read your files, access the network, and do anything else your user account can do. Only install plugins you built yourself or received from a source you trust.

Plugins extend CyberEther with additional blocks. A plugin is distributed as a single `.cep` file, which is a bundle containing the plugin's native libraries and optional example flowgraphs. Once installed, the plugin's blocks show up in the block catalog and behave like any built-in block.

This page covers the user side of plugins. If you want to build your own plugin, see [Creating Plugins](/docs/plugins).

> [!NOTE]
>
> Plugin bundles are experimental. The format and loader behavior can change between CyberEther versions, so a plugin may need to be rebuilt for a newer release.

## Before You Install

Ask yourself these questions before installing any plugin:

- Do you know who made it? Prefer plugins from developers you can identify, with public source code you could inspect.
- Did you download it from the official place? Get the `.cep` file from the developer's own website or repository releases, not from a re-upload somewhere else.
- Does it match what the developer published? If the developer provides a checksum for the file, verify it.

If you cannot answer these questions, do not install the plugin. CyberEther will ask you to confirm that you trust the source during installation, and that confirmation is your responsibility.

## Installing A Plugin

You need a `.cep` file saved somewhere on your computer. Pick a permanent location for it, such as a `Plugins` folder in your home directory, because CyberEther loads the plugin from that exact path every time it starts. Then:

1. Launch CyberEther.
2. Open the **CyberEther** menu in the top menubar and click **Preferences...**. The shortcut is `CTRL+,`.
3. In the **Sections** sidebar of the preferences window, click **Registry**.
4. Find the **Plugins** section and click **Add Plugin**.
5. In the **Register Plugin** dialog, click **Browse File** and pick your `.cep` file. You can also type or paste the full path into the **Plugin Path** field.
6. Read the trust warning. If you trust where the plugin came from, tick **I trust the source of this plugin**. The register button stays disabled until you tick the box.
7. Click **Register Plugin**.

CyberEther loads the plugin immediately and shows a "Plugin loaded." notification. The plugin now appears in the Plugins table with its name, version, and status.

From now on, the plugin loads automatically every time CyberEther starts. If a plugin fails to load at startup, CyberEther logs a warning and continues starting normally.

## Using The Plugin

The plugin's blocks appear in the block catalog alongside the built-in blocks. Add them to a flowgraph the same way you add any other block.

If the plugin bundles example flowgraphs, those are registered too, so you can open them to see the plugin's blocks in a working graph.

## Keep The File In Place

CyberEther remembers the path to the `.cep` file. It does not copy the bundle into its own storage. If you move, rename, or delete the file, the plugin shows a **Missing** status the next time CyberEther starts and its blocks will not be available. To fix that, delete the stale entry and register the plugin again from its new location.

## Managing Installed Plugins

The Plugins table in **Preferences → Registry** lists every registered plugin with its status and two actions:

- **Reload** unloads and reloads that plugin. Use it after replacing the `.cep` file with a newer version. All flowgraphs must be closed first, otherwise CyberEther asks you to close them.
- **Delete** removes the plugin from the registered list. Blocks that were already loaded stay available until you restart CyberEther. The `.cep` file itself is not deleted from your disk.

The **Flowgraph** menu in the top menubar also has a **Reload Plugins** item that reloads every registered plugin at once.

## Troubleshooting

The **Status** column in the Plugins table tells you where a plugin stands:

| Status | Meaning |
|--------|---------|
| Loaded | The plugin is active and its blocks are available. |
| Not loaded | The plugin is registered but failed to load. Check the log for details. |
| Missing | The `.cep` file is no longer at its registered path. |
| Invalid extension | The registered path does not point to a `.cep` file. |

If registration fails with a "Failed to load plugin." notification, the most common causes are:

- **The plugin needs a newer CyberEther.** Every bundle declares a minimum version. Update CyberEther and try again.
- **The bundle has no compatible target.** A bundle ships libraries for specific systems and architectures. A bundle built only for macOS on Apple Silicon will not load on Linux, for example. Ask the developer for a build matching your platform.
- **The plugin was built for a different plugin ABI.** Plugins are experimental, so a plugin built against an older CyberEther may be rejected by a newer one. Ask the developer for a rebuild.

The exact reason is printed to the log. Run CyberEther from a terminal and look for lines starting with `[PLUGIN]`.

A few limitations to be aware of:

- Plugins are not supported in the browser version of CyberEther.
- There is no command line flag for loading plugins. Registration happens through the preferences window and persists across sessions.
