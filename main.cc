#include <thread>
#include <memory>

#include "render/base.hpp"
#include "samurai/samurai.hpp"
#include "jetstream/base.hh"

using namespace Jetstream;

int main() {
    JST_DEBUG("Welcome to Jetstream!");

    Vector<Device::CPU, CF32> buffer(50);

    Block<FFT, Device::CPU>({}, {
        buffer,
    });
}
