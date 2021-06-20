#ifndef JETSTREAM_LPT_CPU_H
#define JETSTREAM_LPT_CPU_H

#include "jetstream/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

class CPU : public Generic  {
public:
    explicit CPU(const Config &);
    ~CPU();

protected:
    Result _compute();
    Result _present();
};

} // namespace Jetstream::Lineplot

#endif
