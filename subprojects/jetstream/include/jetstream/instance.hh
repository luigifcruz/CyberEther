#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <memory>
#include <vector>

#include "jetstream/base.hh"
#include "jetstream/module.hh"

namespace Jetstream {

class JETSTREAM_API Instance {
 public:
    const Result conduit(const std::vector<std::shared_ptr<Module>>& modules);
    const Result present();
    const Result compute();
      
 private:
    std::atomic_flag computeSync{false};
    std::atomic_flag presentSync{false};

    std::vector<std::shared_ptr<Module>> modules;
};

Result Conduit(const std::vector<std::shared_ptr<Module>>&);
Result Compute();
Result Present();

}  // namespace Jetstream

#endif
