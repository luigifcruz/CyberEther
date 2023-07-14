#include "../generic.cc"

namespace Jetstream {

// TODO: Remove in favor of module manifest.
template class Window<Device::Metal, CF64>;
template class Window<Device::Metal, CF32>;  

}  // namespace Jetstream
