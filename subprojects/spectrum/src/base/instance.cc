#include "spectrum/base.hpp"

namespace Spectrum {

template<class T>
std::shared_ptr<T> Instance::Create(Instance::Config& cfg) {
    return std::make_shared<T>(cfg);
};

template<class T>
std::shared_ptr<LinePlot> Instance::create(LinePlot::Config& cfg) {
    auto lineplot = std::make_shared<typename T::LinePlot>(cfg, *state);
    this->cfg.lineplots.push_back(lineplot);
    return lineplot;
}
template std::shared_ptr<FFTW> Instance::Create<FFTW>(Instance::Config &);
template std::shared_ptr<LinePlot> Instance::create<FFTW>(LinePlot::Config&);

} // namespace Render

