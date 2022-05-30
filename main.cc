#include <thread>
#include <memory>

#include "samurai/samurai.hpp"
#include "jetstream/base.hh"

using namespace Jetstream;

int main() {
    JST_DEBUG("Welcome to Jetstream!");

    Backend::Initialize<Device::Metal>({});
    Backend::State<Device::Metal>();

    Render::Initialize<Device::Metal>({});

    Render::Create();
    Render::Destroy();

    Memory::Vector<Device::CPU, CF32> data(2<<20);

    auto win = Block<Window, Device::CPU>({
        .size = data.size(),
    }, {});

    auto mul = Block<Multiply, Device::CPU>({
        .size = data.size(),
    }, {
        .factorA = data, 
        .factorB = win->getWindowBuffer(),
    });

    auto fft = Block<FFT, Device::CPU>({
        .size = data.size(),
    }, {
        .buffer = mul->getProductBuffer(),
    });

    auto amp = Block<Amplitude, Device::CPU>({
        .size = data.size(), 
    }, {
        .buffer = fft->getOutputBuffer(),
    });

    auto scl = Block<Scale, Device::CPU>({
        .size = data.size(),
        .range = {-100.0, 0.0},
    }, {
        .buffer = amp->getOutputBuffer(),
    });

    for (U64 i = 0; i < 512; i++) {
        Jetstream::Compute();
        JST_INFO("Compute {} finished.", i);
    }

    JST_INFO("Successfully finished!");
}
