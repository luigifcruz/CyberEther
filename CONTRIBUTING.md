# Contributing to CyberEther
Contributions are welcome! Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests and invite you to submit pull requests directly in this repository. 

> [!WARNING]
>
> Keep in mind that since this project is still in its early stages, the API is not stable and it is subject to change.

- The code is riddled with `// TODO: ` comments. These are things that need to be done. If you want to help out, look for these comments and try to implement them.
- If you want to add a new feature, please create an issue first. This way we can discuss the feature and make sure it's something we want to add.
- If you want to fix a bug, please create an issue first. This way we can discuss the bug and make sure it's something we want to fix.
- If you are adding a new module, block, dependency, or serialization, look for the appropriate `[NEW ... HOOK]`. These will tell you where to add new `includes` and `defines`. 
    - New module: `[NEW MODULE HOOK]`.
    - New block: `[NEW BLOCK HOOK]`.
    - New dependency: `[NEW DEPENDENCY HOOK]`.
    - New serialization: `[NEW SERIALIZATION HOOK]`.
    - New contributor (you): `[NEW CONTRIBUTOR HOOK]`.
- The library follows the [Google C++ Code Style Guide](https://google.github.io/styleguide/cppguide.html).
- The default line length is 88 but this can be overridden if necessary.
- The default indentation size is 4 spaces. No exceptions.

## Components 

### Testing
To enable building the tests, make sure `catch2` is installed on your system.

### Return Values
All functions should return a `Result` type. This is an enum with the following values:

- `Result::SUCCESS`: The function completed successfully.
- `Result::WARNING`: The function completed with a warning. This is a recoverable error.
- `Result::ERROR`: The function completed with an error. This is a recoverable error.
- `Result::FATAL`: The function completed with a fatal error. This is an unrecoverable error.

### Logger
We use our own logging system. It's a simple wrapper around `fmtlib` with a few extra features. The logging system is thread safe and can be used in a multithreaded environment. The verbose level can be selected at runtime using the `JST_DEBUG` environment variable. The default level if you build for `debugrelease` is `DEBUG`, otherwise it's `INFO`. The levels are as follows:

- `JST_DEBUG=4`: **TRACE** - Very detailed logs, which may include high-volume information such as protocol payloads. *This is compiled out in release builds.*
- `JST_DEBUG=3`: **DEBUG** - Debugging information, less detailed than trace, typically not enabled in release builds.
- `JST_DEBUG=2`: **INFO** - Informational messages, which are normally enabled in release builds.
- `JST_DEBUG=1`: **WARN** - Warning messages, typically for non-critical issues, which can be recovered or which are temporary failures.
- `JST_DEBUG=0`: **ERROR & FATAL** - Error or fatal messages - most of the time these are (hopefully) logged before the application crashes.

If you are writing a new module or block, this is the way to communicate a problem or a warning to the user. If you print something to the console and return a `Result::ERROR`, `Result::WARNING`, or `Result::FATAL`, the compositor will display a notification to the user with the last message you printed.

### Global Defines
Global defines are defined in the `jetstream_config.hh` file.

- `JETSTREAM_VERSION_STR`: The current version of the library.
- `JETSTREAM_BUILD_TYPE`: The current build type. This can be `debug`, `release`, or `debugrelease`.
- `JETSTREAM_BUILD_OPTIMIZATION`: The current optimization level. This can be `0`, `1`, `2`, or `3`.
- `JST_IS_STATIC`: Defined if the current build is static.
- `JST_IS_SHARED`: Defined if the current build is shared.
- `JST_DEBUG_MODE`: Defined if the current build is a debug build.
- `JST_RELEASE_MODE`: Defined if the current build is a release build.
- `JST_OS_[...]`: Flags which operating system the current build was compiled for.
    - `JST_OS_WINDOWS`: Windows (64-bits).
    - `JST_OS_ANDROID`: Android.
    - `JST_OS_IOS`: iOS (iPhone or iPad).
    - `JST_OS_MAC`:macOS.
    - `JST_OS_LINUX`: Linux.
    - `JST_OS_BROWSER`: Browser (Emscripten).

## Licensing and Attribution
CyberEther is licensed under the [MIT License](/LICENSE). All contributions to this project will be released under the MIT License. By submitting a pull request, you are agreeing to comply with this license and for any contributions to be released under it.

Don't forget to add yourself to the [ACKNOWLEDGMENTS](/ACKNOWLEDGMENTS.md) file!

## Code of Conduct
Don't be a jerk. Otherwise, you will be banned from contributing to this project. You will be warned once and then banned if you continue to be a jerk. Please report if you see someone being a jerk on project grounds.