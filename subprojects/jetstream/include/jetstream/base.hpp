#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include "jetstream/type.hpp"
#include "jetstream/module.hpp"

namespace Jetstream {

class Instance {
 public:
    const Result stream(const std::vector<std::shared_ptr<Module>>& modules);
    const Result present();
    const Result compute();
      
 private:
    std::vector<std::shared_ptr<Module>> modules;
    std::atomic_flag computeSync{false};
    std::atomic_flag presentSync{false};
};

template<typename T>
std::shared_ptr<T> Block(const typename T::Config& config, const typename T::Input& input) {
    return std::make_shared<T>(config, input);
}

Result Stream(const std::vector<std::shared_ptr<Module>>&);
Result Compute();
Result Present(); 

} // namespace Jetstream

#endif
