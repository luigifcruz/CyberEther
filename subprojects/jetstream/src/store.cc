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
//
// Modules
//

// Device::CPU

#ifdef JETSTREAM_MODULE_FFT_CPU_AVAILABLE
        { {"fft",           "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<FFT, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_FILTER_CPU_AVAILABLE
        { {"filter",        "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Filter, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WINDOW_CPU_AVAILABLE
        { {"window",        "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Window, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_CPU_AVAILABLE
        { {"multiply",      "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Multiply, Device::CPU, CF32>(r); } },
        { {"multiply",      "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Multiply, Device::CPU,  F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_AMPLITUDE_CPU_AVAILABLE
        { {"amplitude",     "cpu",     "", "CF32",  "F32"}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Amplitude, Device::CPU, CF32, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SCALE_CPU_AVAILABLE
        { {"scale",         "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Scale, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE
        { {"soapy",         "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Soapy, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_AUDIO_CPU_AVAILABLE
        { {"audio",         "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Audio, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE
        { {"lineplot",      "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Lineplot, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE
        { {"waterfall",     "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Waterfall, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE
        { {"spectrogram",   "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Spectrogram, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE
        { {"constellation", "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Constellation, Device::CPU, CF32>(r); } },
#endif

// Device::Metal

#ifdef JETSTREAM_MODULE_FFT_METAL_AVAILABLE
        { {"fft",           "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<FFT, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_FILTER_METAL_AVAILABLE
        { {"filter",        "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Filter, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WINDOW_METAL_AVAILABLE
        { {"window",        "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Window, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
        { {"multiply",      "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Multiply, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_AMPLITUDE_METAL_AVAILABLE
        { {"amplitude",     "metal",     "", "CF32",  "F32"}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Amplitude, Device::Metal, CF32, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SCALE_METAL_AVAILABLE
        { {"scale",         "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Scale, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SOAPY_METAL_AVAILABLE
        { {"soapy",         "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Soapy, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_AUDIO_METAL_AVAILABLE
        // TODO: Add Metal Audio.
#endif
#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
        { {"lineplot",      "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Lineplot, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_METAL_AVAILABLE
        { {"waterfall",     "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Waterfall, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_METAL_AVAILABLE
        { {"spectrogram",   "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Spectrogram, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_METAL_AVAILABLE
        { {"constellation", "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Constellation, Device::Metal, CF32>(r); } },
#endif

//
// Bundles
//

// Device::CPU

#ifdef JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE
        { {"lineplot-ui",      "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Lineplot, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE
        { {"waterfall-ui",     "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Waterfall, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE
        { {"spectrogram-ui",   "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Spectrogram, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE
        { {"constellation-ui", "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Constellation, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE
        { {"soapy-ui",         "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Soapy, Device::CPU, CF32>(r); } },
#endif

// Device::Metal

#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
        { {"lineplot-ui",      "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Lineplot, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_METAL_AVAILABLE
        { {"waterfall-ui",     "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Waterfall, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_METAL_AVAILABLE
        { {"spectrogram-ui",   "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Spectrogram, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_METAL_AVAILABLE
        // TODO: Add Metal Constellation.
#endif
#ifdef JETSTREAM_MODULE_SOAPY_METAL_AVAILABLE
        { {"soapy-ui",         "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { instance.addModule<Bundles::Soapy, Device::Metal, CF32>(r); } },
#endif
    };
    return list;
}

}  // namespace Jetstream