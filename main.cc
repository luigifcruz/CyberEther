#include <thread>
#include <memory>

#include "render/base.hpp"
#include "samurai/samurai.hpp"
#include "jetstream/base.hh"

using namespace Jetstream;

int main() {
    JST_DEBUG("Welcome to Jetstream!");

    Vector<Device::CPU, CF32> data(2<<20);

    auto win = Block<Window, Device::CPU>({
        data.size(),
    }, {});

    auto mul = Block<Multiply, Device::CPU>({
        data.size(),
    }, {
        data, 
        win->getWindowBuffer(),
    });

    auto fft = Block<FFT, Device::CPU>({
        data.size(),
    }, {
        mul->getProductBuffer(),
    });

    auto amp = Block<Amplitude, Device::CPU>({
       data.size(), 
    }, {
        fft->getOutputBuffer(),
    });

    auto scl = Block<Scale, Device::CPU>({
        data.size(),
        {-100.0, 0.f},
    }, {
        amp->getOutputBuffer(),
    });

    Jetstream::Conduit({
        win, 
        mul,
        fft, 
        amp,
        scl,
    });

    for (U64 i = 0; i < 512; i++) {
        Jetstream::Compute();
        JST_INFO("Compute {} finished.", i);
    }

    JST_INFO("Successfully ran!");
}
