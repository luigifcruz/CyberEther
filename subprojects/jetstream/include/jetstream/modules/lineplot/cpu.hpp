#ifndef JETSTREAM_LPT_CPU_H
#define JETSTREAM_LPT_CPU_H

#include "jetstream/modules/lineplot/generic.hpp"

namespace Jetstream {

class Lineplot::CPU : public Lineplot  {
public:
    explicit CPU(const Config & cfg, Connections& input);
    ~CPU();

protected:
    Result _compute();
    Result _present();
};

} // namespace Jetstream

#endif
