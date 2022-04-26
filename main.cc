#include <thread>
#include <memory>

#include "render/base.hpp"
#include "samurai/samurai.hpp"
#include "jetstream/base.hh"

using namespace Jetstream;

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    JST_DEBUG("Jetstream debug.");

    Vector<Device::CPU, CF32> buffer;

    FFT<Device::CPU>::Config config {
        .direction = FFT<Device::CPU>::Direction::Forward,
    };

    FFT<Device::CPU>::Input input {
        .buffer = buffer,
    };

    FFT<Device::CPU> fft(config, input);

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
