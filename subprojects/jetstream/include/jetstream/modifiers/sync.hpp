#ifndef JETSTREAM_SYNC_H
#define JETSTREAM_SYNC_H

#include "jetstream/module.hpp"

namespace Jetstream {

template<typename T>
class Sync : public T {
public:
    Sync(const typename T::Config& config, const typename T::Input& input) :
        T(config, input)
    {}
};

} // namespace Jetstream

#endif
