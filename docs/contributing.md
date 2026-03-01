---
title: Contributing
description: Guidelines for contributing to CyberEther.
order: 10
category: Resources
---

Contributions are welcome! Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests and invite you to submit pull requests directly in this repository.

> [!WARNING]
>
> Keep in mind that since this project is still in its early stages, the API is not stable and it is subject to change.

## Getting Started

The code is riddled with `// TODO:` comments. These are things that need to be done and are a good place to start if you want to help out.

Before starting work on a new feature or bug fix, please create an issue first. This allows us to discuss the change and ensure it aligns with the project's direction.

### Code Hooks

When adding new modules, blocks, dependencies, or serializations, look for the appropriate hook comments in the codebase. These indicate where to add new `includes` and `defines`.

| Hook | Purpose |
|------|---------|
| `[NEW MODULE HOOK]` | Adding a new module |
| `[NEW BLOCK HOOK]` | Adding a new block |
| `[NEW DEPENDENCY HOOK]` | Adding a new dependency |
| `[NEW SERIALIZATION HOOK]` | Adding a new serialization |
| `[NEW CONTRIBUTOR HOOK]` | Adding yourself as a contributor |

## Code Style

The library follows the [Google C++ Code Style Guide](https://google.github.io/styleguide/cppguide.html) with the following modifications:

- Default line length is 88 characters (can be overridden if necessary).
- Default indentation is 4 spaces. No exceptions.

## API Reference

### Return Values

All functions should return a `Result` type. This is an enum with the following values:

| Result | Description |
|--------|-------------|
| <span style="color: #22c55e; font-weight: 600">Result::SUCCESS</span> | Completed successfully. |
| <span style="color: #eab308; font-weight: 600">Result::WARNING</span> | Recoverable warning. |
| <span style="color: #f97316; font-weight: 600">Result::ERROR</span> | Recoverable error. |
| <span style="color: #ef4444; font-weight: 600">Result::FATAL</span> | Unrecoverable error. |

### Logger

We use our own logging system. It's a simple wrapper around `fmtlib` with a few extra features. The logging system is thread-safe and can be used in a multithreaded environment.

The verbose level can be selected at runtime using the `JST_DEBUG` environment variable. The default level for `debugrelease` builds is `DEBUG`, otherwise it's `INFO`.

| Level | Name | Description |
|-------|------|-------------|
| `JST_DEBUG=4` | <span style="color: #888; font-weight: 600">TRACE</span> | Very detailed logs, which may include high-volume information such as protocol payloads. *Compiled out in release builds.* |
| `JST_DEBUG=3` | <span style="color: #3b82f6; font-weight: 600">DEBUG</span> | Debugging information, less detailed than trace, typically not enabled in release builds. |
| `JST_DEBUG=2` | <span style="color: #22c55e; font-weight: 600">INFO</span> | Informational messages, which are normally enabled in release builds. |
| `JST_DEBUG=1` | <span style="color: #eab308; font-weight: 600">WARN</span> | Warning messages, typically for non-critical issues, which can be recovered or which are temporary failures. |
| `JST_DEBUG=0` | <span style="color: #ef4444; font-weight: 600">ERROR & FATAL</span> | Error or fatal messages - most of the time these are (hopefully) logged before the application crashes. |

If you are writing a new module or block, use the logger to communicate problems or warnings to the user. If you print something to the console and return a `Result::ERROR`, `Result::WARNING`, or `Result::FATAL`, the compositor will display a notification to the user with the last message you printed.

### Testing

To enable building the tests, make sure `catch2` is installed on your system.

## Global Defines

Global defines are preprocessor macros defined in the `jetstream_config.hh` file. These are automatically generated during the build process and provide information about the build configuration and target platform.

### Build Configuration

These defines provide information about the current build setup.

| Define | Description |
|--------|-------------|
| `JETSTREAM_VERSION_STR` | Current version of the library. |
| `JETSTREAM_BUILD_TYPE` | Build type: `debug`, `release`, or `debugrelease`. |
| `JETSTREAM_BUILD_OPTIMIZATION` | Optimization level: `0`, `1`, `2`, or `3`. |
| `JST_IS_STATIC` | Defined if the build is static. |
| `JST_IS_SHARED` | Defined if the build is shared. |
| `JST_DEBUG_MODE` | Defined if the build is debug. |
| `JST_RELEASE_MODE` | Defined if the build is release. |

### Platform Flags

One of the following `JST_OS_*` flags will be defined based on the target platform.

| Flag | Platform |
|------|----------|
| `JST_OS_WINDOWS` | Windows (64-bit) |
| `JST_OS_MAC` | macOS |
| `JST_OS_LINUX` | Linux |
| `JST_OS_ANDROID` | Android |
| `JST_OS_IOS` | iOS (iPhone, iPad, tvOS) |
| `JST_OS_BROWSER` | Browser (Emscripten) |

## Licensing and Attribution

CyberEther is licensed under the [MIT License](/LICENSE). All contributions to this project will be released under the MIT License. By submitting a pull request, you are agreeing to comply with this license and for any contributions to be released under it.

Don't forget to add yourself to the [ACKNOWLEDGMENTS](/ACKNOWLEDGMENTS.md) file!

## Code of Conduct

Don't be a jerk. Otherwise, you will be banned from contributing to this project. You will be warned once and then banned if you continue to be a jerk. Please report if you see someone being a jerk on project grounds.
