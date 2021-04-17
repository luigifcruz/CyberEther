#ifndef SPECTRUM_BASE_INSTANCE_H
#define SPECTRUM_BASE_INSTANCE_H

#include "spectrum/types.hpp"

namespace Spectrum {

class Instance {
public:
    struct Config {
        std::shared_ptr<Render::Instance> render;
        int *bandwidth;
        int *frequency;
        size_t *size;
        void *buffer;
        DataFormat format;
    };

    Instance(Config& c) : cfg(c) {};
    virtual ~Instance() = default;

protected:
    Config& cfg;
};

} // namespace Spectrum

#endif
