#include "../generic.cc"

namespace Jetstream {

template class Window<Device::CPU, CF64>;
template class Window<Device::CPU, CF32>;  

}  // namespace Jetstream

template class Window<Device::Metal, CF64>;
template class Window<Device::Metal, CF32>;
