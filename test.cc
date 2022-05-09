#include <thread>
#include <memory>
#include <iostream>

enum Device {
    CPU,
    CUDA
};

template<Device D> class InstanceImp;

class Instance {
 public:
    struct Config {
    };

    template<Device D> 
    static std::shared_ptr<Instance> Factory(const Config& config) {
        return std::make_shared<InstanceImp<D>>(config);
    }
};

template<>
class InstanceImp<Device::CPU> : public Instance {
 public:
    InstanceImp(const Config& config) {
        std::cout << "CPU" << std::endl;
    }
};

template<class T>
inline void Create(std::shared_ptr<T>& member, const auto& config) {
    member = T::template Factory<Device::CPU>(config);
}

int main() {
    std::shared_ptr<Instance> instance;
    Instance::Config config;
    Create(instance, config);
}