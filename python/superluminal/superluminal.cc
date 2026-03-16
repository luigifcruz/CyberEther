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

template <DeviceType D>
struct ToNanobind;

template <>
struct ToNanobind<DeviceType::CPU> {
    using value_type = nb::device::cpu;
};

template <>
struct ToNanobind<DeviceType::CUDA> {
    using value_type = nb::device::cuda;
};

template<DeviceType D, typename T>
Tensor numpy_to_tensor(nb::handle handle) {
    // Cast to array.

    using Type = nb::ndarray<nb::numpy, T, nb::c_contig, typename ToNanobind<D>::value_type>;
    auto array = nb::cast<Type>(handle);

    // Extract shape from numpy array

    std::vector<U64> shape;
    for (size_t i = 0; i < array.ndim(); ++i) {
        shape.push_back(static_cast<U64>(array.shape(i)));
    }

    if constexpr (D == DeviceType::CUDA) {
        throw std::invalid_argument("CUDA ndarray buffers are not supported by this binding yet");
    }

    Tensor tensor(static_cast<void*>(array.data()), D, TypeToDataType<T>(), shape);

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

    nb::enum_<DeviceType>(m, "device")
        .value("none", DeviceType::None)
        .value("cpu", DeviceType::CPU)
        .value("cuda", DeviceType::CUDA)
        .value("metal", DeviceType::Metal)
        .value("vulkan", DeviceType::Vulkan)
        .value("webgpu", DeviceType::WebGPU);

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
                    self.buffer = numpy_to_tensor<DeviceType::CPU, CF32>(input);
                    return;
                } else if (nb::isinstance<nb::ndarray<nb::numpy, F32, nb::c_contig, nb::device::cpu>>(input)) {
                    self.buffer = numpy_to_tensor<DeviceType::CPU, F32>(input);
                    return;
                }
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
                if (nb::isinstance<nb::ndarray<nb::numpy, CF32, nb::c_contig, nb::device::cuda>>(input)) {
                    self.buffer = numpy_to_tensor<DeviceType::CUDA, CF32>(input);
                    return;
                } else if (nb::isinstance<nb::ndarray<nb::numpy, F32, nb::c_contig, nb::device::cuda>>(input)) {
                    self.buffer = numpy_to_tensor<DeviceType::CUDA, F32>(input);
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
        .def_rw("device", &Superluminal::InstanceConfig::device)
        .def_rw("remote", &Superluminal::InstanceConfig::remote)
        .def_rw("remote_broker", &Superluminal::InstanceConfig::remoteBroker)
        .def_rw("remote_codec", &Superluminal::InstanceConfig::remoteCodec)
        .def_rw("remote_encoder", &Superluminal::InstanceConfig::remoteEncoder)
        .def_rw("remote_auto_join", &Superluminal::InstanceConfig::remoteAutoJoin)
        .def_rw("remote_framerate", &Superluminal::InstanceConfig::remoteFramerate)
        .def_rw("preferred_device", &Superluminal::InstanceConfig::preferredDevice);

    m.def("initialize", &Superluminal::Initialize, nb::arg("config") = Superluminal::InstanceConfig());
    m.def("start", &Superluminal::Start);
    m.def("stop", &Superluminal::Stop);
    m.def("presenting", &Superluminal::Presenting);
    m.def("update", &Superluminal::Update, nb::arg("name") = std::string());
    m.def("show", &Superluminal::Show);
    m.def("poll_events", &Superluminal::PollEvents, nb::arg("wait") = true);
    m.def("plot", &Superluminal::Plot);
    m.def("remote_room_id", &Superluminal::RemoteRoomId);
    m.def("remote_invite_url", &Superluminal::RemoteInviteUrl);
    m.def("remote_access_token", &Superluminal::RemoteAccessToken);
    m.def("print_remote_info", &Superluminal::PrintRemoteInfo);

    static std::vector<std::shared_ptr<nb::object>> stored_callbacks;

    m.def("box", [](const std::string& title, const Superluminal::Mosaic& mosaic, nb::object callback) {
        stored_callbacks.push_back(std::make_shared<nb::object>(std::move(callback)));
        return Superluminal::Box(title, mosaic, [&]() {
            if (Py_IsInitialized()) {
                nb::gil_scoped_acquire gil;
                (*stored_callbacks.back())();
            }
        });
    }, nb::arg("title"), nb::arg("mosaic"), nb::arg("callback"));

    m.def("terminate", []() {
        stored_callbacks.clear();
        Superluminal::Terminate();
    });

    m.def("text", [](const std::string& text) {
        return Superluminal::Text("{}", text);
    }, nb::arg("text"));

    m.def("slider", [](const std::string& label, F32 min, F32 max, nb::list value_list) {
        if (value_list.size() != 1) {
            throw std::invalid_argument("Slider value list must contain exactly one element");
        }
        F32 value = nb::cast<F32>(value_list[0]);
        auto result = Superluminal::Slider(label, min, max, value);
        value_list[0] = value;
        return result;
    }, nb::arg("label"), nb::arg("min"), nb::arg("max"), nb::arg("value"));

    m.def("markdown", &Superluminal::Markdown, nb::arg("content"));

    m.def("mosaic_layout", &Superluminal::MosaicLayout);
}
