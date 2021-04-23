#ifndef SPECTRUM_BASE_INSTANCE_H
#define SPECTRUM_BASE_INSTANCE_H

#include "spectrum/types.hpp"
#include "lineplot.hpp"

namespace Spectrum {

class Instance {
public:
    struct Config {
        std::shared_ptr<Render::Instance> render;
        double bandwidth;
        double frequency;
        size_t size;
        void* buffer;
        DataFormat format;
        std::vector<std::shared_ptr<LinePlot>> lineplots;
    };

    Instance(Config& c) : cfg(c) {};
    virtual ~Instance() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;

    virtual Result feed() = 0;

    Config& config() {
        return cfg;
    }

    virtual std::shared_ptr<LinePlot> create(LinePlot::Config&) = 0;

protected:
    Config& cfg;
};

} // namespace Spectrum

#endif
