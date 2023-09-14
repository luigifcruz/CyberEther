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
    moduleList = defaultModuleList();
}

// TODO: Make this case-unsensitive.
// TODO: Cache last query.
ModuleListStore& Store::_moduleList(const std::string& filter) {
    if (filter.empty()) {
        return moduleList;
    }

    filteredModuleList.clear();

    for (const auto& [key, value] : moduleList) {
        if (value.title.find(filter) != std::string::npos ||
            value.small.find(filter) != std::string::npos ||
            value.detailed.find(filter) != std::string::npos) {
            filteredModuleList[key] = value;
        }
    }

    return filteredModuleList;
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
        { {"fft",               "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<FFT, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_FILTER_CPU_AVAILABLE
        { {"filter",            "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Filter, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WINDOW_CPU_AVAILABLE
        { {"window",            "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Window, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_CPU_AVAILABLE
        { {"multiply",          "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Multiply, Device::CPU,  F32>(r); } },
        { {"multiply",          "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Multiply, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_AMPLITUDE_CPU_AVAILABLE
        { {"amplitude",         "cpu",     "", "CF32",  "F32"}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Amplitude, Device::CPU, CF32, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SCALE_CPU_AVAILABLE
        { {"scale",             "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Scale, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE
        { {"soapy",             "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Soapy, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_AUDIO_CPU_AVAILABLE
        { {"audio",             "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Audio, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_FM_CPU_AVAILABLE
        { {"fm",                "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<FM, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_CONSTANT_CPU_AVAILABLE
        { {"multiply-constant", "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<MultiplyConstant, Device::CPU,  F32>(r); } },
        { {"multiply-constant", "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<MultiplyConstant, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE
        { {"lineplot",          "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Lineplot, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE
        { {"waterfall",         "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Waterfall, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE
        { {"spectrogram",       "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Spectrogram, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE
        { {"constellation",     "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Constellation, Device::CPU, CF32>(r); } },
#endif

// Device::Metal

#ifdef JETSTREAM_MODULE_FFT_METAL_AVAILABLE
        { {"fft",               "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<FFT, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_FILTER_METAL_AVAILABLE
        { {"filter",            "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Filter, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WINDOW_METAL_AVAILABLE
        { {"window",            "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Window, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
        { {"multiply",          "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Multiply, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_AMPLITUDE_METAL_AVAILABLE
        { {"amplitude",         "metal",     "", "CF32",  "F32"}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Amplitude, Device::Metal, CF32, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SCALE_METAL_AVAILABLE
        { {"scale",             "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Scale, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SOAPY_METAL_AVAILABLE
        { {"soapy",             "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Soapy, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_AUDIO_METAL_AVAILABLE
        { {"audio",             "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Audio, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_CONSTANT_METAL_AVAILABLE
        { {"multiply-constant", "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<MultiplyConstant, Device::Metal,  F32>(r); } },
        { {"multiply-constant", "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<MultiplyConstant, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
        { {"lineplot",          "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Lineplot, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_METAL_AVAILABLE
        { {"waterfall",         "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Waterfall, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_METAL_AVAILABLE
        { {"spectrogram",       "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Spectrogram, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_METAL_AVAILABLE
        { {"constellation",     "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Constellation, Device::Metal, CF32>(r); } },
#endif

//
// Bundles
//

// Device::CPU

#ifdef JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE
        { {"lineplot-view",      "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Lineplot, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE
        { {"waterfall-view",     "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Waterfall, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE
        { {"spectrogram-view",   "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Spectrogram, Device::CPU, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE
        { {"constellation-view", "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Constellation, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE
        { {"soapy-view",         "cpu", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Soapy, Device::CPU, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SCALE_CPU_AVAILABLE
        { {"scale-view",         "cpu",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Scale, Device::CPU, F32>(r); } },
#endif

// Device::Metal

#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
        { {"lineplot-view",      "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Lineplot, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_METAL_AVAILABLE
        { {"waterfall-view",     "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Waterfall, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_METAL_AVAILABLE
        { {"spectrogram-view",   "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Spectrogram, Device::Metal, F32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_METAL_AVAILABLE
        // TODO: Add Metal Constellation.
#endif
#ifdef JETSTREAM_MODULE_SOAPY_METAL_AVAILABLE
        { {"soapy-view",         "metal", "CF32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Soapy, Device::Metal, CF32>(r); } },
#endif
#ifdef JETSTREAM_MODULE_SCALE_METAL_AVAILABLE
        { {"scale-view",         "metal",  "F32",     "",     ""}, [](Instance& instance, Parser::ModuleRecord& r) { return instance.addModule<Bundles::Scale, Device::Metal, F32>(r); } },
#endif
    };
    return list;
}

ModuleListStore& Store::defaultModuleList() {
    static ModuleListStore list = {
        //
        // Modules
        //

        {"fft",
            {
                false,
                "FFT",
                "Transforms input data to frequency domain.",
                "Fast Fourier Transform that converts time-domain data to its frequency components. Supports real and complex data types.", 
                {
                    {
#ifdef JETSTREAM_MODULE_FFT_CPU_AVAILABLE
                        {Device::CPU, {{"CF32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_FFT_METAL_AVAILABLE
                        {Device::Metal, {{"CF32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"filter",
            {
                false,
                "Filter",
                "Generates a FIR bandpass filter taps.",
                "The Filter module creates Finite Impulse Response (FIR) bandpass filter coefficients (taps) based on specified frequency parameters. These taps can be used to filter input data, attenuating or amplifying certain frequency components.",
                {
                    {
#ifdef JETSTREAM_MODULE_FILTER_CPU_AVAILABLE
                        {Device::CPU, {{"CF32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_FILTER_METAL_AVAILABLE
                        {Device::Metal, {{"CF32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"window",
            {
                false,
                "Window",
                "Generates a hanning window taps.",
                "The Window module produces coefficients (taps) for a Hanning window based on the specified size. Hanning windows are commonly used in signal processing to reduce the side lobes and spectral leakage when performing operations like the Fourier Transform.",
                {
                    {
#ifdef JETSTREAM_MODULE_WINDOW_CPU_AVAILABLE
                        {Device::CPU, {{"CF32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_WINDOW_METAL_AVAILABLE
                        {Device::Metal, {{"CF32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"multiply",
            {
                false,
                "Multiply",
                "Element-wise multiplication.",
                "Takes 'factorA' and 'factorB' as inputs and outputs the result as 'product'.",
                {
                    {
#ifdef JETSTREAM_MODULE_MULTIPLY_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}, {"CF32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
                        {Device::Metal, {{"CF32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"amplitude",
            {
                false,
                "Amplitude",
                "Computes amplitude from complex data.",
                "Takes complex input data and calculates the amplitude or magnitude for each data point.",
                {
                    {
#ifdef JETSTREAM_MODULE_AMPLITUDE_CPU_AVAILABLE
                        {Device::CPU, {{"", "CF32", "F32"}}},
#endif
#ifdef JETSTREAM_MODULE_AMPLITUDE_METAL_AVAILABLE
                        {Device::Metal, {{"", "CF32", "F32"}}},
#endif
                    }
                }
            }
        },
        {"scale",
            {
                false,
                "Scale",
                "Scales input data by a factor.",
                "Multiplies each data point in the input by a specified scaling factor, adjusting its magnitude.",
                {
                    {
#ifdef JETSTREAM_MODULE_SCALE_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_SCALE_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"soapy",
            {
                false,
                "Soapy",
                "Interface for SoapySDR devices.",
                "Provides an interface to communicate and control SoapySDR supported devices, facilitating data acquisition and device configuration.",
                {
                    {
#ifdef JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE
                        {Device::CPU, {{"CF32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_SOAPY_METAL_AVAILABLE
                        {Device::Metal, {{"CF32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"audio",
            {
                false,
                "Audio",
                "Outputs an audio stream to using the sound card.",
                "Handles audio data for playback, recording, or further processing. Supports various audio formats and rates.",
                {
                    {
#ifdef JETSTREAM_MODULE_AUDIO_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_AUDIO_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"fm",
            {
                false,
                "FM",
                "Demodulates Frequency Modulation signal.",
                "The FM module processes an input Frequency Modulated (FM) signal and extracts the original baseband signal. This demodulation is essential for decoding information transmitted over FM radio waves or other FM communication systems.",
                {
                    {
#ifdef JETSTREAM_MODULE_FM_CPU_AVAILABLE
                        {Device::CPU, {{"CF32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"multiply-constant",
            {
                false,
                "Multiply Constant",
                "Multiplies input by a constant.",
                "Takes input data and multiplies each data point by a specified constant value. Suitable for signal amplitude adjustments.",
                {
                    {
#ifdef JETSTREAM_MODULE_MULTIPLY_CONSTANT_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}, {"CF32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_CONSTANT_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}, {"CF32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"lineplot",
            {
                false,
                "Lineplot",
                "Displays data in a line plot.",
                "Visualizes input data in a line graph format, suitable for time-domain signals and waveform displays.",
                {
                    {
#ifdef JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"waterfall",
            {
                false,
                "Waterfall",
                "Displays data in a waterfall plot.",
                "Visualizes frequency-domain data over time in a 2D color-coded format. Suitable for spectral analysis.",
                {
                    {
#ifdef JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"spectrogram",
            {
                false,
                "Spectrogram",
                "Displays a spectrogram of data.",
                "Visualizes how frequencies of input data change over time. Represents amplitude of frequencies using color.",
                {
                    {
#ifdef JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"constellation",
            {
                false,
                "Constellation",
                "Displays a constellation plot.",
                "Visualizes modulated data in a 2D scatter plot. Commonly used in digital communication to represent symbol modulation.",
                { 
                    {
#ifdef JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE
                        {Device::CPU, {{"CF32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_METAL_AVAILABLE
                        {Device::Metal, {{"CF32", "", ""}}},
#endif
                    }
                }
            }
        },

        //
        // Bundles
        //

        {"lineplot-view",
            {
                true,
                "Lineplot View",
                "Displays data in a line plot with extra controls.",
                "Visualizes input data in a line graph format, suitable for time-domain signals and waveform displays.",
                {
                    {
#ifdef JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"waterfall-view",
            {
                true,
                "Waterfall View",
                "Displays data in a waterfall plot with extra controls.",
                "Visualizes frequency-domain data over time in a 2D color-coded format. Suitable for spectral analysis.",
                {
                    {
#ifdef JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"spectrogram-view",
            {
                true,
                "Spectrogram View",
                "Displays a spectrogram of data with extra controls.",
                "Visualizes how frequencies of input data change over time. Represents amplitude of frequencies using color.",
                {
                    {
#ifdef JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"constellation-view",
            {
                true,
                "Constellation View",
                "Displays a constellation plot with extra controls.",
                "Visualizes modulated data in a 2D scatter plot. Commonly used in digital communication to represent symbol modulation.",
                { 
                    {
#ifdef JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE
                        {Device::CPU, {{"CF32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_CONSTELLATION_METAL_AVAILABLE
                        {Device::Metal, {{"CF32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"soapy-view",
            {
                true,
                "Soapy View",
                "Interface for SoapySDR devices with extra controls.",
                "Provides an interface to communicate and control SoapySDR supported devices, facilitating data acquisition and device configuration.",
                {
                    {
#ifdef JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE
                        {Device::CPU, {{"CF32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_SOAPY_METAL_AVAILABLE
                        {Device::Metal, {{"CF32", "", ""}}},
#endif
                    }
                }
            }
        },
        {"scale-view",
            {
                false,
                "Scale View",
                "Scales input data by a factor with extra controls.",
                "Multiplies each data point in the input by a specified scaling factor, adjusting its magnitude.",
                {
                    {
#ifdef JETSTREAM_MODULE_SCALE_CPU_AVAILABLE
                        {Device::CPU, {{"F32", "", ""}}},
#endif
#ifdef JETSTREAM_MODULE_SCALE_METAL_AVAILABLE
                        {Device::Metal, {{"F32", "", ""}}},
#endif
                    }
                }
            }
        },
    };
    return list;
};

}  // namespace Jetstream
