#ifndef JETSTREAM_INTERFACE_HH
#define JETSTREAM_INTERFACE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"

namespace Jetstream {

class JETSTREAM_API Interface {
 public:
    virtual constexpr Device device() const = 0;
    virtual constexpr std::string name() const = 0;
    virtual constexpr std::string prettyName() const = 0;

    virtual Result drawNodeControl() {
        return Result::SUCCESS;
    }

    virtual Result drawNode() {
        return Result::SUCCESS;
    }

    virtual Result drawView() {
        return Result::SUCCESS;
    }
    
    virtual Result drawControl() {
        return Result::SUCCESS;
    }

    virtual Result drawInfo() {
        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif
