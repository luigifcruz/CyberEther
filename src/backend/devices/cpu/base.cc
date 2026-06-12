#include "jetstream/backend/devices/cpu/base.hh"

#include "jetstream/logger.hh"

#include <thread>

namespace Jetstream::Backend {

CPU::CPU(const Config& config) : config(config) {
    const auto hardwareThreads = std::thread::hardware_concurrency();
    const auto hardwareThreadsText = hardwareThreads == 0 ? std::string("Unknown") : std::to_string(hardwareThreads);
    const auto pythonRuntimePath = config.pythonRuntimePath.empty() ? std::string("Auto") : config.pythonRuntimePath;

    // Print device information.

    JST_INFO("-----------------------------------------------------");
    JST_INFO("Jetstream Heterogeneous Backend [CPU]")
    JST_INFO("-----------------------------------------------------");
    JST_INFO("Hardware Threads:   {}", hardwareThreadsText);
    JST_INFO("Python Runtime:     {}", pythonRuntimePath);
    JST_INFO("-----------------------------------------------------");
}

const std::string& CPU::getPythonRuntimePath() const {
    return config.pythonRuntimePath;
}

}  // namespace Jetstream::Backend
