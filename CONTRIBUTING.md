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

## Licensing and Attribution
CyberEther is licensed under the [MIT License](/LICENSE). All contributions to this project will be released under the MIT License. By submitting a pull request, you are agreeing to comply with this license and for any contributions to be released under it.

Don't forget to add yourself to the [ACKNOWLEDGMENTS](/ACKNOWLEDGMENTS.md) file!

## Code of Conduct
Don't be a jerk. Otherwise, you will be banned from contributing to this project. You will be warned once and then banned if you continue to be a jerk. Please report if you see someone being a jerk on project grounds.