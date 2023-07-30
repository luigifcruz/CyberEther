#include "jetstream/store.hh"

#include "jetstream/bundles/base.hh"
#include "jetstream/bundles/constellation.hh"
#include "jetstream/modules/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/viewport/base.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream {

Store& Store::GetInstance() {
    static Store store;
    return store;
}

Store::Store() {
    modules = defaultModules();
    viewports = defaultViewports();
    backends = defaultBackends();
    renders = defaultRenders();
}

BackendStore& Store::defaultBackends() {
    static BackendStore list = {
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        { {"metal"},  [](Instance& instance, Parser::BackendRecord& r) { return instance.buildBackend<Device::Metal>(r); } },
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        { {"vulkan"}, [](Instance& instance, Parser::BackendRecord& r) { return instance.buildBackend<Device::Vulkan>(r); } },
#endif
#ifdef JETSTREAM_BACKEND_WEBGPU_AVAILABLE
        { {"webgpu"}, [](Instance& instance, Parser::BackendRecord& r) { return instance.buildBackend<Device::WebGPU>(r); } },
#endif
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        { {"cpu"},    [](Instance& instance, Parser::BackendRecord& r) { return instance.buildBackend<Device::CPU>(r);    } },
#endif
    };
    return list;
}

ViewportStore& Store::defaultViewports() {
    static ViewportStore list = {
#ifdef JETSTREAM_VIEWPORT_GLFW_AVAILABLE
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
        { {"metal", "glfw"},  [](Instance& instance, Parser::ViewportRecord& r) { return instance.buildViewport<Viewport::GLFW<Device::Metal>>(r);  } },
#endif
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
        { {"vulkan", "glfw"}, [](Instance& instance, Parser::ViewportRecord& r) { return instance.buildViewport<Viewport::GLFW<Device::Vulkan>>(r); } },
#endif
#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
        { {"webgpu", "glfw"}, [](Instance& instance, Parser::ViewportRecord& r) { return instance.buildViewport<Viewport::GLFW<Device::WebGPU>>(r); } },
#endif
#endif
    };
    return list;
}

RenderStore& Store::defaultRenders() {
    static RenderStore list = {
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
        { {"metal"},  [](Instance& instance, Parser::RenderRecord& r) { return instance.buildRender<Device::Metal>(r);  } },
#endif
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
        { {"vulkan"}, [](Instance& instance, Parser::RenderRecord& r) { return instance.buildRender<Device::Vulkan>(r); } },
#endif
#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
        { {"webgpu"}, [](Instance& instance, Parser::RenderRecord& r) { return instance.buildRender<Device::WebGPU>(r); } },
#endif
    };
    return list;
}

ModuleStore& Store::defaultModules() {
    static ModuleStore list = {
        { {"soapy",     "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Soapy,     Device::CPU, CF32>(r); } },
        { {"multiply",  "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Multiply,  Device::CPU, CF32>(r); } },
        { {"filter",    "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Filter,    Device::CPU, CF32>(r); } },
        { {"fft",       "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<FFT,       Device::CPU, CF32>(r); } },
        { {"window",    "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Window,    Device::CPU, CF32>(r); } },
        { {"amplitude", "cpu",     "", "CF32",  "F32"}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Amplitude, Device::CPU, CF32,  F32>(r); } },
        { {"scale",     "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Scale,     Device::CPU,  F32>(r); } },

        { {"constellation-ui", "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Constellation, Device::CPU, CF32>(r); } },
        { {"lineplot-ui",      "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Lineplot,      Device::CPU,  F32>(r); } },
        { {"spectrogram-ui",   "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Spectrogram,   Device::CPU,  F32>(r); } },
        { {"waterfall-ui",     "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Waterfall,     Device::CPU,  F32>(r); } },
        { {"soapy-ui",         "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Soapy,         Device::CPU, CF32>(r); } },

        // TODO: Add ifdefs.
    };
    return list;
}

}  // namespace Jetstream