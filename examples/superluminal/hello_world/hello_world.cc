#include <jetstream/superluminal.hh>

using namespace Jetstream;

int main() {
    std::cout << "Welcome to Superluminal!" << std::endl;

    auto data = Tensor<Device::CPU, CF32>({1, 8192});

    for (U64 j = 0; j < data.shape(1); j++) {
        data[{0, j}] = std::rand() % 3;
    }

    Superluminal::Plot("Random", {{1}}, {
        .buffer = data,
        .type = Superluminal::Type::Line,
        .source = Superluminal::Domain::Time,
        .display = Superluminal::Domain::Frequency,
    });

    Superluminal::Show();

    std::cout << "Goodbye from Superluminal!" << std::endl;

    return 0;
}
