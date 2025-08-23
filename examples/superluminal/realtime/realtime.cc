#include <jetstream/superluminal.hh>

using namespace Jetstream;

int main() {
    std::cout << "Welcome to Superluminal!" << std::endl;

    auto data = Tensor<Device::CPU, CF32>({1, 8192});

    Superluminal::Plot("Sine", {{1}}, {
        .buffer = data,
        .type = Superluminal::Type::Line,
    });

    Superluminal::RealtimeLoop([&](const bool& running){
        F32 phase = 0.0f;

        while (running) {
            // Generate sine wave.

            for (U64 i = 0; i < data.shape(0); i++) {
                for (U64 j = 0; j < data.shape(1); j++) {
                    data[{i, j}] = std::sin(j * 0.01 + phase) * 20.0 + 30.0;
                }
            }

            phase += 0.1f;

            Superluminal::Update();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    std::cout << "Goodbye from Superluminal!" << std::endl;

    return 0;
}
