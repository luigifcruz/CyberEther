#include <iostream>

enum Device {
    CPU, 
    CUDA,
    GENERIC,
};

template<Device D> class Lineplot;

template<>
class Lineplot<Device::GENERIC> {
 protected:
    int x = 42;
};

template<>
class Lineplot<Device::CPU> : public Lineplot<Device::GENERIC> {
 public:
    Lineplot() {
        std::cout << "YO " << this->x << std::endl;
    }
};

int main() {
    Lineplot<Device::GENERIC> lpt;

    lpt = Lineplot<Device::CPU>();

    return 0;
}
