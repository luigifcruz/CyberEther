#include <jetstream/superluminal.hh>

using namespace Jetstream;

static Result App() {
    std::cout << "Welcome to Superluminal!" << std::endl;

    Tensor data(DeviceType::CPU, TypeToDataType<F32>(), {1, 8192});

    JST_CHECK(Superluminal::Plot("Sine", {{1}}, {
        .buffer = data,
        .type = Superluminal::Type::Line,
        .options = {},
    }));

    JST_CHECK(Superluminal::RealtimeLoop([&](const bool& running){
        F32 phase = 0.0f;

        while (running) {
            // Generate sine wave.

            for (U64 i = 0; i < data.shape(0); i++) {
                for (U64 j = 0; j < data.shape(1); j++) {
                    data.at<F32>(i, j) = std::sin(j * 0.01f + phase) * 0.25f;
                }
            }

            phase += 0.1f;

            Superluminal::Update();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }));

    std::cout << "Goodbye from Superluminal!" << std::endl;

    return Result::SUCCESS;
}

int main() {
    return App() == Result::SUCCESS ? 0 : 1;
}
