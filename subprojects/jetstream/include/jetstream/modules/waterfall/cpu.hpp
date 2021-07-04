#ifndef JETSTREAM_WTF_CPU_H
#define JETSTREAM_WTF_CPU_H

#include "jetstream/modules/waterfall/generic.hpp"

namespace Jetstream {

class Waterfall::CPU : public Waterfall  {
public:
    explicit CPU(const Config & cfg, Connections& input);
    ~CPU();

protected:
    Result _compute();

    std::vector<float> bin;
};

} // namespace Jetstream

#endif
