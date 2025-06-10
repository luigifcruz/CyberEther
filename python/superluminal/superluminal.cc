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

template<Device D, typename T>
Tensor<D, T> numpy_to_tensor(nb::ndarray<nb::numpy, T, nb::c_contig, nb::device::cpu> array) {
    auto pointer = static_cast<T*>(array.data());
    
    return Tensor<D, T>(pointer, {1, 8192});
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

    nb::enum_<Result>(m, "result")
        .value("success", Result::SUCCESS)
        .value("warning", Result::WARNING)
        .value("error", Result::ERROR)
        .value("fatal", Result::FATAL);

    nb::class_<Superluminal::PlotConfig>(m, "plot_config")
        .def(nb::init<>())
        .def_prop_rw("buffer",
            [](Superluminal::PlotConfig &self) {
                return self.buffer;
            },
            [](Superluminal::PlotConfig &self, nb::ndarray<nb::numpy, CF32, nb::c_contig, nb::device::cpu> array) {
                self.buffer = numpy_to_tensor<Device::CPU, CF32>(array);
            })
        .def_rw("type", &Superluminal::PlotConfig::type)
        .def_rw("batchAxis", &Superluminal::PlotConfig::batchAxis)
        .def_rw("channelAxis", &Superluminal::PlotConfig::channelAxis)
        .def_rw("channelIndex", &Superluminal::PlotConfig::channelIndex)
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
        .def_rw("headless", &Superluminal::InstanceConfig::headless)
        .def_rw("endpoint", &Superluminal::InstanceConfig::endpoint);

    m.def("initialize", &Superluminal::Initialize, nb::arg("config") = Superluminal::InstanceConfig());
    m.def("terminate", &Superluminal::Terminate);
    m.def("start", &Superluminal::Start);
    m.def("stop", &Superluminal::Stop);
    m.def("presenting", &Superluminal::Presenting);
    m.def("update", &Superluminal::Update, nb::arg("name") = std::string());
    m.def("block", &Superluminal::Block);
    m.def("poll_events", &Superluminal::PollEvents, nb::arg("wait") = true);
    m.def("plot", &Superluminal::Plot);
}