---
title: Contributing
description: Guidelines, code style, logging conventions, and hooks for contributing to CyberEther.
order: 370
category: Development
---

Contributions of all sizes are welcome! Pull requests are the best way to propose improvements, bug fixes, new blocks, and documentation updates directly in the repository.

> [!WARNING]
> CyberEther is still evolving rapidly. Internal APIs are not yet frozen and may change between releases.

## Before You Start

1. **Open an issue first** before investing significant time in a new feature or architectural fix. A quick discussion ensures your work aligns with the project's direction.
2. **Look for `// TODO:` comments** if you want a concrete place to jump in. The codebase has plenty of well-scoped tasks waiting to be picked up.
3. **Set up your build and read the core guides** first: [Building From Source](/docs/building), [Architecture](/docs/architecture), and [Blocks And Modules](/docs/blocks-and-modules) will help you get oriented quickly.

### Finding Integration Points With Code Hooks

When adding a new module, block, dependency, serialization type, or contributor entry, search the repository for these hook comments to find every place that needs an `include`, `define`, or entry:

| Hook Comment | When to Search for It |
|---|---|
| `[NEW MODULE HOOK]` | Adding a new compute module |
| `[NEW BLOCK HOOK]` | Adding a new flowgraph block |
| `[NEW DEPENDENCY HOOK]` | Adding a new third-party dependency |
| `[NEW SERIALIZATION HOOK]` | Adding a new serializable type |
| `[NEW CONTRIBUTOR HOOK]` | Adding yourself to the Acknowledgments page |

## Code Style

CyberEther follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with two local rules:

- **Line length:** 88 characters by default (may be exceeded when breaking hurts readability).
- **Indentation:** 4 spaces strictly, with no exceptions.

## Core Conventions

### Return Values

Functions that can fail should return `Result`, an enum representing the outcome:

| Value | Meaning |
|---|---|
| `Result::SUCCESS` | Completed without issues. |
| `Result::WARNING` | Completed with a recoverable warning. |
| `Result::ERROR` | Recoverable failure. |
| `Result::FATAL` | Unrecoverable failure. |

### Logging

CyberEther uses a thread-safe logging wrapper built around `fmtlib`. Set the `JST_DEBUG` environment variable at runtime to control verbosity. The default level is `DEBUG` when `JST_DEBUG_MODE` is defined (such as `debug` and `debugoptimized` builds) and `INFO` in `release` builds.

| Setting | Level | When It Is Used |
|---|---|---|
| `JST_DEBUG=4` | **TRACE** | High-volume diagnostic output such as protocol payloads. Compiled out in release builds. |
| `JST_DEBUG=3` | **DEBUG** | Detailed debugging messages, off by default in release builds unless enabled with `JST_DEBUG=3` or `-v`. |
| `JST_DEBUG=2` | **INFO** | Routine informational logs enabled in release builds. |
| `JST_DEBUG=1` | **WARN** | Non-critical, recoverable issues or temporary failures. |
| `JST_DEBUG=0` | **ERROR & FATAL** | Serious errors and fatal conditions logged before aborting. |

Colors turn off automatically when standard output is not a compatible terminal. You can also disable colored logs explicitly via `NO_COLOR`:

```bash
NO_COLOR=1 cyberether
```

Applications embedding Jetstream can disable log colors programmatically by calling `JST_LOG_COLOR(false)`.

> [!TIP]
> When writing a block or module, log a clear message right before returning `Result::WARNING`, `Result::ERROR`, or `Result::FATAL`. The compositor automatically surfaces your last log message as a user-facing notification.

### Running Tests and Build Macros

For full instructions on running Catch2, shader parser, browser, and opt-in SDR hardware tests, plus the list of generated `jetstream/config.hh` macros, see [Building From Source](/docs/building#running-tests).

## Licensing and Attribution

CyberEther is released under the [MIT License](/docs/license). By submitting a pull request, you agree that your contributions are licensed under the same MIT terms.

Remember to add your name under the `[NEW CONTRIBUTOR HOOK]` on the [Acknowledgments](/docs/acknowledgments) page!

## Code of Conduct

Don't be a jerk. Anyone behaving abusively receives one warning and will be banned from contributing if it continues. Please report any unacceptable behavior you witness in project spaces.
