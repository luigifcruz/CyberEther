#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <jetstream/superluminal.hh>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Jetstream;

template <Device D>
struct ToNanobind;

template <>
struct ToNanobind<Device::CPU> {
    using value_type = nb::device::cpu;
};

template <>
struct ToNanobind<Device::CUDA> {
    using value_type = nb::device::cuda;
};

template<Device D, typename T>
Tensor<D, T> numpy_to_tensor(nb::handle handle) {
    // Cast to array.

    using Type = nb::ndarray<nb::numpy, T, nb::c_contig, typename ToNanobind<D>::value_type>;
    auto array = nb::cast<Type>(handle);

    // Extract shape from numpy array

    std::vector<U64> shape;
    for (size_t i = 0; i < array.ndim(); ++i) {
        shape.push_back(static_cast<U64>(array.shape(i)));
    }

    // Create zero-copy tensor.

    auto pointer = static_cast<T*>(array.data());
    auto tensor = Tensor<D, T>(pointer, shape);

    // Set tensor hash.

    tensor.set_hash(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer)));

    return tensor;
}

NB_MODULE(_impl, m) {
    nb::class_<Superluminal> superluminal(m, "impl");

    nb::enum_<Superluminal::Type>(m, "type")
        .value("line", Superluminal::Type::Line)
        .value("heat", Superluminal::Type::Heat)
        .value("scatter", Superluminal::Type::Scatter)
        .value("waterfall", Superluminal::Type::Waterfall);

    nb::enum_<Superluminal::Domain>(m, "domain")
        .value("time", Superluminal::Domain::Time)
        .value("frequency", Superluminal::Domain::Frequency);

    nb::enum_<Superluminal::Operation>(m, "operation")
        .value("real", Superluminal::Operation::Real)
        .value("imaginary", Superluminal::Operation::Imaginary)
        .value("amplitude", Superluminal::Operation::Amplitude)
        .value("phase", Superluminal::Operation::Phase);

    nb::enum_<Device>(m, "device")
        .value("none", Device::None)
        .value("cpu", Device::CPU)
        .value("cuda", Device::CUDA)
        .value("metal", Device::Metal)
        .value("vulkan", Device::Vulkan)
        .value("webgpu", Device::WebGPU);

    nb::enum_<Result>(m, "result")
        .value("success", Result::SUCCESS)
        .value("warning", Result::WARNING)
        .value("error", Result::ERROR)
        .value("fatal", Result::FATAL);

    nb::class_<Superluminal::PlotConfig>(m, "plot_config")
        .def(nb::init<>())
        .def_prop_rw("buffer",
            [](Superluminal::PlotConfig &self) { return self.buffer; },
            [](Superluminal::PlotConfig &self, nb::handle input) {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
                if (nb::isinstance<nb::ndarray<nb::numpy, CF32, nb::c_contig, nb::device::cpu>>(input)) {
                    self.buffer = numpy_to_tensor<Device::CPU, CF32>(input);
                    return;
                } else if (nb::isinstance<nb::ndarray<nb::numpy, F32, nb::c_contig, nb::device::cpu>>(input)) {
                    self.buffer = numpy_to_tensor<Device::CPU, F32>(input);
                    return;
                }
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
                if (nb::isinstance<nb::ndarray<nb::numpy, CF32, nb::c_contig, nb::device::cuda>>(input)) {
                    self.buffer = numpy_to_tensor<Device::CUDA, CF32>(input);
                    return;
                } else if (nb::isinstance<nb::ndarray<nb::numpy, F32, nb::c_contig, nb::device::cuda>>(input)) {
                    self.buffer = numpy_to_tensor<Device::CUDA, F32>(input);
                    return;
                }
#endif
                throw std::invalid_argument("Unsupported buffer type or device");
            },
            "Set the buffer")
        .def_rw("type", &Superluminal::PlotConfig::type)
        .def_rw("batch_axis", &Superluminal::PlotConfig::batchAxis)
        .def_rw("channel_axis", &Superluminal::PlotConfig::channelAxis)
        .def_rw("channel_index", &Superluminal::PlotConfig::channelIndex)
        .def_rw("source", &Superluminal::PlotConfig::source)
        .def_rw("display", &Superluminal::PlotConfig::display)
        .def_rw("operation", &Superluminal::PlotConfig::operation)
        .def_rw("options", &Superluminal::PlotConfig::options);

    nb::class_<Superluminal::InstanceConfig>(m, "instance_config")
        .def(nb::init<>())
        .def_rw("device_id", &Superluminal::InstanceConfig::deviceId)
        .def_rw("interface_scale", &Superluminal::InstanceConfig::interfaceScale)
        .def_rw("interface_size", &Superluminal::InstanceConfig::interfaceSize)
        .def_rw("window_title", &Superluminal::InstanceConfig::windowTitle)
        .def_rw("remote", &Superluminal::InstanceConfig::remote)
        .def_rw("preferred_device", &Superluminal::InstanceConfig::preferredDevice);

    m.def("initialize", &Superluminal::Initialize, nb::arg("config") = Superluminal::InstanceConfig());
    m.def("terminate", &Superluminal::Terminate);
    m.def("start", &Superluminal::Start);
    m.def("stop", &Superluminal::Stop);
    m.def("presenting", &Superluminal::Presenting);
    m.def("update", &Superluminal::Update, nb::arg("name") = std::string());
    m.def("block", &Superluminal::Block);
    m.def("poll_events", &Superluminal::PollEvents, nb::arg("wait") = true);
    m.def("plot", &Superluminal::Plot);

    m.def("mosaic_layout", &Superluminal::MosaicLayout);
}
