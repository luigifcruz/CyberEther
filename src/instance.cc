#include "jetstream/instance.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"
#include "jetstream/config.hh"
#include "jetstream/registry.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/components/font.hh"

#include "resources/fonts/compressed_jbmm.hh"

#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <deque>
#include <thread>
#include <atomic>

namespace Jetstream {

struct Instance::Impl {
    bool created = false;
    bool started = false;
    bool computing = false;
    bool presenting = false;

    DeviceType device;
    std::shared_mutex flowgraphsMutex;
    std::unordered_map<std::string, std::shared_ptr<Flowgraph>> flowgraphs;

    std::shared_ptr<Viewport::Generic> viewport;
    std::shared_ptr<Render::Window> render;
    std::shared_ptr<Compositor> compositor;

    std::shared_ptr<Instance::Remote> remote;
};

Instance::Instance() {
    impl = std::make_unique<Impl>();
}

Instance::~Instance() {
    impl.reset();
}

Result Instance::create(const Config& config) {
    JST_INFO("[INSTANCE] Creating instance.");
    JST_ASSERT(!impl->created, "[INSTANCE] Instance already created.");

    // Choose optimal device.

    std::deque<DeviceType> priority = {
        DeviceType::Metal,
        DeviceType::Vulkan,
        DeviceType::WebGPU,
    };

    if (config.device.has_value()) {
        priority.push_front(config.device.value());
    }

    // Initialize viewport, renderer and compositor.

    {
        auto backendConfig = Backend::Config {
            .headless = config.headless,
        };
        auto viewportConfig = Viewport::Config {
            .size = config.size,
            .framerate = config.framerate,
        };
        auto renderConfig = Render::Window::Config {
            .scale = config.scale,
        };

        auto buildGlfw = [&]<DeviceType D>() -> Result {
            JST_CHECK(Backend::Initialize<D>(backendConfig));

            auto viewport = std::make_shared<Viewport::GLFW<D>>(viewportConfig);
            JST_CHECK(viewport->create());

            auto render = std::make_shared<Render::WindowImp<D>>(renderConfig, viewport);
            JST_CHECK(render->create());

            impl->viewport = std::move(viewport);
            impl->render = std::move(render);

            return Result::SUCCESS;
        };

        auto buildHeadless = [&]<DeviceType D>() -> Result {
            JST_CHECK(Backend::Initialize<D>(backendConfig));

            auto viewport = std::make_shared<Viewport::Headless<D>>(viewportConfig);
            JST_CHECK(viewport->create());

            auto render = std::make_shared<Render::WindowImp<D>>(renderConfig, viewport);
            JST_CHECK(render->create());

            impl->viewport = std::move(viewport);
            impl->render = std::move(render);

            return Result::SUCCESS;
        };

        std::optional<Result> result;

        for (const auto& device : priority) {
            if (config.headless) {
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
                if (device == DeviceType::Vulkan) {
                    result = buildHeadless.template operator()<DeviceType::Vulkan>();
                    break;
                }
#endif
            } else {
#ifdef JETSTREAM_VIEWPORT_GLFW_AVAILABLE
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
                if (device == DeviceType::Metal) {
                    result = buildGlfw.template operator()<DeviceType::Metal>();
                    break;
                }
#endif
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
                if (device == DeviceType::Vulkan) {
                    result = buildGlfw.template operator()<DeviceType::Vulkan>();
                    break;
                }
#endif
#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
                if (device == DeviceType::WebGPU) {
                    result = buildGlfw.template operator()<DeviceType::WebGPU>();
                    break;
                }
#endif
#endif  // JETSTREAM_VIEWPORT_GLFW_AVAILABLE
            }
        }

        if (!result.has_value()) {
            JST_ERROR("[INSTANCE] No viewport backend available.");
            return Result::ERROR;
        }

        JST_CHECK(result.value());
    }

    if (config.compositor.has_value()) {
        impl->compositor = std::make_shared<Compositor>(config.compositor.value());
        JST_CHECK(impl->compositor->create(shared_from_this(), impl->render, impl->viewport));
    }

    // Load default fonts.

    {
        std::shared_ptr<Render::Components::Font> font;

        Render::Components::Font::Config cfg;
        cfg.data = jbmm_compressed_data;
        cfg.size = 32.0f;

        JST_CHECK(impl->render->build(font, cfg));
        JST_CHECK(impl->render->addFont("default_mono", font));
    }

    impl->remote = std::make_shared<Remote>(impl->viewport.get());

    impl->created = true;
    return Result::SUCCESS;
}

Result Instance::destroy() {
    JST_INFO("[INSTANCE] Destroying instance.");
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");

    for (const auto& [_, flowgraph] : impl->flowgraphs) {
        JST_CHECK(flowgraph->destroy());
    }

    if (impl->remote && impl->remote->started()) {
        JST_CHECK(impl->remote->destroy());
    }

    if (impl->compositor) {
        JST_CHECK(impl->compositor->destroy());
    }

    // Unload default fonts.

    if (impl->render && impl->render->hasFont("default_mono")) {
        JST_CHECK(impl->render->removeFont("default_mono"));
    }

    // Destroy render and viewport resources.

    if (impl->render) {
        JST_CHECK(impl->render->destroy());
        impl->render.reset();
    }

    if (impl->viewport) {
        JST_CHECK(impl->viewport->destroy());
        impl->viewport.reset();
    }

    impl->created = false;
    return Result::SUCCESS;
}

Result Instance::start() {
    JST_INFO("[INSTANCE] Starting instance.");
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");
    JST_ASSERT(!impl->started, "[INSTANCE] Instance already started.");

    JST_CHECK(impl->render->start());

    impl->started = true;
    impl->computing = true;
    impl->presenting = true;

    return Result::SUCCESS;
}

Result Instance::stop() {
    JST_INFO("[INSTANCE] Stopping instance.");
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");
    JST_ASSERT(impl->started, "[INSTANCE] Instance not started.");

    JST_CHECK(impl->render->stop());

    impl->started = false;
    impl->computing = false;
    impl->presenting = false;

    return Result::SUCCESS;
}

bool Instance::computing() const {
    return impl->computing;
}

Result Instance::compute() {
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");
    JST_ASSERT(impl->started, "[INSTANCE] Instance not started.");

    // Copy flowgraph pointers while holding lock, then release before compute.

    std::vector<std::shared_ptr<Flowgraph>> flowgraphs;
    {
        std::shared_lock lock(impl->flowgraphsMutex);
        flowgraphs.reserve(impl->flowgraphs.size());
        for (const auto& [_, flowgraph] : impl->flowgraphs) {
            flowgraphs.push_back(flowgraph);
        }
    }

    for (const auto& flowgraph : flowgraphs) {
        JST_CHECK(flowgraph->compute());
    }

    return Result::SUCCESS;
}

bool Instance::presenting() const {
    return impl->presenting;
}

Result Instance::present(const std::function<Result()>& callback) {
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");
    JST_ASSERT(impl->started, "[INSTANCE] Instance not started.");

    // Create new render frame.

    const auto& beginRes = impl->render->begin();
    if (beginRes == Result::SKIP) {
        return Result::SUCCESS;
    }
    if (beginRes != Result::SUCCESS) {
        impl->presenting = false;
        return beginRes;
    }

    // Update the modules present logic.

    std::vector<std::shared_ptr<Flowgraph>> flowgraphs;
    {
        std::shared_lock lock(impl->flowgraphsMutex);
        flowgraphs.reserve(impl->flowgraphs.size());
        for (const auto& [_, flowgraph] : impl->flowgraphs) {
            flowgraphs.push_back(flowgraph);
        }
    }

    for (const auto& flowgraph : flowgraphs) {
        JST_CHECK(flowgraph->present());
    }

    // Render the inferface via compositor and callback.

    if (callback) {
        JST_CHECK(callback());
    }

    if (impl->compositor) {
        JST_CHECK(impl->compositor->present());
    }

    // Finish the render frame.

    const auto& endRes = impl->render->end();
    if (endRes == Result::SKIP) {
        return Result::SUCCESS;
    }
    if (endRes != Result::SUCCESS) {
        impl->presenting = false;
        return endRes;
    }

    // Capture frame.

    if (impl->remote && impl->remote->started()) {
        impl->remote->captureFrame();
    }

    // Process compositor interactions.

    if (impl->compositor) {
        JST_CHECK(impl->compositor->poll());
    }

    return Result::SUCCESS;
}

bool Instance::polling() const {
    return computing() && presenting() && impl->viewport->keepRunning();
}

Result Instance::poll(const bool wait) {
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");
    JST_ASSERT(impl->started, "[INSTANCE] Instance not started.");

    if (wait) {
        JST_CHECK(impl->viewport->waitEvents());
    } else {
        JST_CHECK(impl->viewport->pollEvents());
    }

    return Result::SUCCESS;
}

const std::shared_ptr<Instance::Remote>& Instance::remote() {
    return impl->remote;
}

Result Instance::flowgraphCreate(const std::string name,
                                 const Flowgraph::Config& config,
                                 std::shared_ptr<Flowgraph>& flowgraph) {
    JST_INFO("[INSTANCE] Creating flowgraph.");
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");

    {
        std::unique_lock lock(impl->flowgraphsMutex);
        if (impl->flowgraphs.contains(name)) {
            JST_ERROR("[INSTANCE] Flowgraph name '{}' already exists.", name);
            return Result::ERROR;
        }
    }

    flowgraph = std::make_shared<Flowgraph>();
    JST_CHECK(flowgraph->create(config, shared_from_this(), impl->render, impl->compositor));

    {
        std::unique_lock lock(impl->flowgraphsMutex);
        impl->flowgraphs.emplace(name, flowgraph);
    }

    return Result::SUCCESS;
}

Result Instance::flowgraphDestroy(const std::string name) {
    JST_INFO("[INSTANCE] Destroying flowgraph.");
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");

    std::shared_ptr<Flowgraph> fg;

    {
        std::unique_lock lock(impl->flowgraphsMutex);
        if (!impl->flowgraphs.contains(name)) {
            JST_ERROR("[INSTANCE] Flowgraph '{}' does not exist.", name);
            return Result::ERROR;
        }
        fg = impl->flowgraphs.at(name);
        impl->flowgraphs.erase(name);
    }

    JST_CHECK(fg->destroy());

    return Result::SUCCESS;
}

Result Instance::flowgraphList(std::unordered_map<std::string, std::shared_ptr<Flowgraph>>& flowgraphs) {
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");

    {
        std::shared_lock lock(impl->flowgraphsMutex);
        flowgraphs.clear();
        for (const auto& [name, fg] : impl->flowgraphs) {
            flowgraphs[name] = fg;
        }
    }

    return Result::SUCCESS;
}

Result Instance::compositorGet(std::shared_ptr<Compositor>& compositor) {
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");

    compositor = impl->compositor;
    return Result::SUCCESS;
}

Result Instance::viewportGet(std::shared_ptr<Viewport::Generic>& viewport) {
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");

    viewport = impl->viewport;
    return Result::SUCCESS;
}

Result Instance::renderGet(std::shared_ptr<Render::Window>& render) {
    JST_ASSERT(impl->created, "[INSTANCE] Instance not created.");

    render = impl->render;
    return Result::SUCCESS;
}

}  // namespace Jetstream
