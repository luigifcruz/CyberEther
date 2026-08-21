#include <jetstream/superluminal.hh>
#include <cmath>
#include <cstdlib>

using namespace Jetstream;

static constexpr float kPi = JST_PI;

void GenerateTestSignal(Tensor& data, float frequency, float amplitude, float noise_level) {
    const float sample_rate = 44100.0f;

    for (U64 j = 0; j < data.shape(1); j++) {
        float t = j / sample_rate;

        float signal = amplitude * std::cos(2.0f * kPi * frequency * t);
        float noise = (std::rand() / float(RAND_MAX) - 0.5f) * noise_level;

        data.at<CF32>(0, j) = CF32(signal + noise, 0.0f);
    }
}

/**
 * Jetstream Superluminal Global Interface Demo
 *
 * Unlike Interface and Box, which occupy a mosaic cell, GlobalInterface draws
 * above the mosaic. The space it takes is reserved, so the plots below shrink
 * to fit underneath it instead of being covered.
 */
static Result App() {
    Tensor data(DeviceType::CPU, TypeToDataType<CF32>(), {1, 8192});

    float frequency = 1000.0f;
    float amplitude = 1.0f;
    float noise_level = 0.1f;

    GenerateTestSignal(data, frequency, amplitude, noise_level);

    // Control bar spanning the top of the window, outside of the mosaic.
    JST_CHECK(Superluminal::GlobalInterface([&]{
        Superluminal::Text("Signal Parameters");
        Superluminal::Slider("Frequency (Hz)", 100.0f, 5000.0f, frequency);
        Superluminal::Slider("Amplitude", 0.1f, 2.0f, amplitude);
        Superluminal::Slider("Noise Level", 0.0f, 0.5f, noise_level);
    }));

    // Both cells resize to fit below the control bar.
    JST_CHECK(Superluminal::Plot("Time", {{1}, {0}}, {
        .buffer = data,
        .type = Superluminal::Type::Line,
        .source = Superluminal::Domain::Time,
        .display = Superluminal::Domain::Time,
        .options = {},
    }));

    JST_CHECK(Superluminal::Plot("Frequency", {{0}, {1}}, {
        .buffer = data,
        .type = Superluminal::Type::Line,
        .source = Superluminal::Domain::Time,
        .display = Superluminal::Domain::Frequency,
        .options = {},
    }));

    JST_CHECK(Superluminal::RealtimeLoop([&](const bool& running){
        while (running) {
            GenerateTestSignal(data, frequency, amplitude, noise_level);

            Superluminal::Update();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }));

    return Result::SUCCESS;
}

int main() {
    return App() == Result::SUCCESS ? 0 : 1;
}
