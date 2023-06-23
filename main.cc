#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;
    
    // Initialize Backends.
    if (Backend::Initialize<Device::CPU>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize CPU backend.");
        return 1;
    }
    JST_INFO("Done");

    if (Backend::Initialize<Device::WebGPU>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize WebGPU backend.");
        return 1;
    }
    JST_INFO("Done");

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
