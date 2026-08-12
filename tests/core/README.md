# Core Tests

The core suite covers all generic CyberEther behavior that is not specific to a production block or module implementation. Generic Block and Module lifecycle tests use synthetic test-owned implementations.

| Meson test ID | Build target | Folder |
|---|---|---|
| `core-application` | `jetstream-core-application-tests` | `application/` |
| `core-compositor` | `jetstream-core-compositor-tests` | `compositor/` |
| `core-extensions` | `jetstream-core-extensions-tests` | `extensions/` |
| `core-flowgraph` | `jetstream-core-flowgraph-tests` | `flowgraph/` |
| `core-integration` | `jetstream-core-integration-tests` | `integration/` |
| `core-lifecycle` | `jetstream-core-lifecycle-tests` | `lifecycle/` |
| `core-memory` | `jetstream-core-memory-tests` | `memory/` |
| `core-platform` | `jetstream-core-platform-tests` | `platform/` |
| `core-registry` | `jetstream-core-registry-tests` | `registry/` |
| `core-render` | `jetstream-core-render-tests` | `render/` |
| `core-runtime` | `jetstream-core-runtime-tests` | `runtime/` |
| `core-serialization` | `jetstream-core-serialization-tests` | `serialization/` |
| `core-settings` | `jetstream-core-settings-tests` | `settings/` |
| `core-tools` | `jetstream-core-tools-tests` | `tools/` |

`core-integration` is not registered on iOS, Android, or Windows.

Configure once, then build every core target through the aggregate alias. Do not start one build per target or source:

```sh
meson setup -Dbuildtype=debugoptimized build
meson compile -C build jetstream-core-tests
meson test -C build --suite core --no-rebuild --num-processes 1 --print-errorlogs
```

Build and run one domain when diagnosing a focused failure:

```sh
meson compile -C build jetstream-core-flowgraph-tests
meson test -C build core-flowgraph --no-rebuild --num-processes 1 --print-errorlogs
```

Some assertions intentionally describe desired behavior that the current implementation violates. They remain active regression failures and have a short defect comment immediately above the assertion.

TODO integration coverage requiring dedicated environments:

- Real window and GPU-backed `Instance` lifecycle across Metal, Vulkan, and WebGPU.
- Browser OPFS, Emscripten main-loop, and main-thread proxy behavior.
- Live GStreamer/WebRTC broker sessions and hardware encoders.
- Successful cross-platform plugin DSO load, unload, and reload.
- Permanently blocking source cancellation and adversarial scheduler concurrency under TSAN.
