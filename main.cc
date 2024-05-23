#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

int main(int argc, char* argv[]) {
    // Parse command line arguments.

    Backend::Config backendConfig;
    Viewport::Config viewportConfig;
    Render::Window::Config renderConfig;
    std::string flowgraphPath;
    Device prefferedBackend = Device::None;

    for (int i = 1; i < argc; i++) {
        const std::string arg = std::string(argv[i]);

        if (arg == "--headless") {
            backendConfig.headless = true;

            continue;
        }

        if (arg == "--endpoint") {
            if (i + 1 < argc) {
                viewportConfig.endpoint = argv[++i];
            }

            continue;
        }

        if (arg == "--backend") {
            // TODO: Add check for valid backend.

            if (i + 1 < argc) {
                prefferedBackend = StringToDevice(argv[++i]);
            }

            continue;
        }

        if (arg == "--no-validation") {
            backendConfig.validationEnabled = false;

            continue;
        }

        if (arg == "--no-vsync") {
            viewportConfig.vsync = false;

            continue;
        }

        if (arg == "--no-hw-acceleration") {
            viewportConfig.hardwareAcceleration = false;

            continue;
        }

        if (arg == "--benchmark") {
            // TODO: Add check for valid output type.

            std::string outputType = "markdown";

            if (i + 1 < argc) {
                outputType = std::string(argv[++i]);
            }

            Benchmark::Run(outputType);

            return 0;
        }

        if (arg == "--framerate") {
            if (i + 1 < argc) {
                viewportConfig.framerate = std::stoul(argv[++i]);
            }

            continue;
        }

        if (arg == "--multisampling") {
            if (i + 1 < argc) {
                backendConfig.multisampling = std::stoul(argv[++i]);
            }

            continue;
        }

        if (arg == "--size") {
            if (i + 2 < argc) {
                viewportConfig.size.width = std::stoul(argv[++i]);
                viewportConfig.size.height = std::stoul(argv[++i]);
            }

            continue;
        }

        if (arg == "--codec") {
            // TODO: Add check for valid codec.

            if (i + 1 < argc) {
                viewportConfig.codec = Render::StringToVideoCodec(argv[++i]);
            }

            continue;
        }

        if (arg == "--device-id") {
            if (i + 1 < argc) {
                backendConfig.deviceId = std::stoul(argv[++i]);
            }

            continue;
        }

        if (arg == "--staging-buffer") {
            if (i + 1 < argc) {
                backendConfig.stagingBufferSize = std::stoul(argv[++i])*1024*1024;
            }

            continue;
        }

        if (arg == "--scale") {
            if (i + 1 < argc) {
                renderConfig.scale = std::stof(argv[++i]);
            }

            continue;
        }

        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options] [flowgraph]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --headless              Enable headless mode." << std::endl;
            std::cout << "  --endpoint [endpoint]   Set the endpoint of the headless viewport (`1.1.1.1:8000`, `./vid.mp4`, etc). Default: `/tmp/cyberether`" << std::endl;
            std::cout << "  --backend [backend]     Set the preferred backend (`Metal`, `Vulkan`, or `WebGPU`)." << std::endl;
            std::cout << "  --framerate [value]     Set the framerate of the headless viewport (FPS). Default: `60`" << std::endl;
            std::cout << "  --multisampling [value] Set the multisampling anti-aliasing factor (`1`, `4`, or `8`). Default: `4`" << std::endl;
            std::cout << "  --codec [codec]         Set the video codec of the headless viewport. Default: `FFV1`" << std::endl;
            std::cout << "  --size [width] [height] Set the initial size of the viewport. Default: `1920 1080`" << std::endl;
            std::cout << "  --scale [scale]         Set the scale of the render window. Default: `1.0`" << std::endl;
            std::cout << "  --benchmark [type]      Run the benchmark and output the results (`markdown`, `json`, or `csv`). Default: `markdown`" << std::endl;
            std::cout << "  --no-hw-acceleration    Disable hardware acceleration. Enabled otherwise." << std::endl;
            std::cout << "Other Options:" << std::endl;
            std::cout << "  --staging-buffer [size] Set the staging buffer size (MB). Default: `64`" << std::endl;
            std::cout << "  --device-id [id]        Set the physical device ID. Default: `0`" << std::endl;
            std::cout << "  --no-validation         Disable Vulkan validation layers. Enabled otherwise." << std::endl;
            std::cout << "  --no-vsync              Disable vsync. Enabled otherwise." << std::endl;
            std::cout << "Other:" << std::endl;
            std::cout << "  --help, -h              Print this help message." << std::endl;
            std::cout << "  --version, -v           Print the version." << std::endl;
            return 0;
        }

        if (arg == "--version" || arg == "-v") {
            std::cout << "CyberEther v" << JETSTREAM_VERSION_STR << "-" << JETSTREAM_BUILD_TYPE << std::endl;
            return 0;
        }

        flowgraphPath = arg;
    }

    // Instance creation.

    Instance instance;

    // Configure instance.

    Instance::Config config = {
        .preferredDevice = prefferedBackend,
        .enableCompositor = true,
        .backendConfig = backendConfig,
        .viewportConfig = viewportConfig,
        .renderConfig = renderConfig
    };

    JST_CHECK_THROW(instance.build(config));

    // Load flowgraph if provided.

    if (!flowgraphPath.empty()) {
        JST_CHECK_THROW(instance.flowgraph().create(flowgraphPath));
    }

    // Start instance.

    instance.start();

    // Start compute thread.

    auto computeThread = std::thread([&]{
        while (instance.computing()) {
            JST_CHECK_THROW(instance.compute());
        }
    });

    // Start graphical thread.

    auto graphicalThreadLoop = [](void* arg) {
        Instance* instance = reinterpret_cast<Instance*>(arg);

        if (instance->begin() == Result::SKIP) {
            return;
        }
        JST_CHECK_THROW(instance->present());
        if (instance->end() == Result::SKIP) {
            return;
        }
    };

#ifdef JST_OS_BROWSER
    emscripten_set_main_loop_arg(graphicalThreadLoop, &instance, 0, 1);
#else
    auto graphicalThread = std::thread([&]{
        while (instance.presenting()) {
            graphicalThreadLoop(&instance);
        }
    });
#endif

    // Start input polling.

#ifdef JST_OS_BROWSER
    emscripten_runtime_keepalive_push();
#else
    while (instance.viewport().keepRunning()) {
        instance.viewport().pollEvents();
    }
#endif

    // Stop instance and wait for threads.

    instance.reset();
    instance.stop();

    if (computeThread.joinable()) {
        computeThread.join();
    }

#ifdef JST_OS_BROWSER
    emscripten_cancel_main_loop();
#else
    if (graphicalThread.joinable()) {
        graphicalThread.join();
    }
#endif

    // Destroy instance.

    instance.destroy();

    // Destroy backend.

    Backend::DestroyAll();

    return 0;
}
