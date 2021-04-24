#ifndef SPECTRUM_FFTW_INSTANCE_H
#define SPECTRUM_FFTW_INSTANCE_H

#include "spectrum/base/instance.hpp"

namespace Spectrum {

class FFTW : public Spectrum::Instance {
public:
    class LinePlot;

    FFTW(Config& c) : Spectrum::Instance(c) {};

    Result create();
    Result destroy();

    Result feed();

    std::shared_ptr<Spectrum::LinePlot> create(Spectrum::LinePlot::Config&);

protected:
    std::complex<float>* fft_out;
    fftwf_plan fft_plan;
};

} // namespace Spectrum

#endif
