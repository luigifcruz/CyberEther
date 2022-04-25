#include <iostream>

enum Device {
    CPU, 
    CUDA,
};

template<Device D>
class Lineplot {
 public:
    Lineplot() {
        std::cout << "GENERIC" << std::endl;
    }

 private:
    struct CUDA {

    };
};

template<>
Lineplot<Device::CPU>::Lineplot() {
    std::cout << "CPU" << std::endl;
}

int main() {
    Lineplot<Device::CPU> lpt;

    lpt = Lineplot<Device::CPU>();

    return 0;
}
