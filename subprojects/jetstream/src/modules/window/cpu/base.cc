#include "../generic.cc"

namespace Jetstream {

// TODO: Remove in favor of module manifest.
template class Window<Device::CPU, CF64>;
template class Window<Device::CPU, CF32>;  

}  // namespace Jetstream
