#include <jetstream/superluminal.hh>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace Jetstream;

/**
 * Generates a test signal combining a sine wave with noise
 * This demonstrates typical signal processing scenarios
 */
void GenerateTestSignal(auto& data, float frequency = 1000.0f, float amplitude = 1.0f, float noise_level = 0.1f) {
    const float sample_rate = 44100.0f;

    for (U64 j = 0; j < data.shape(1); j++) {
        float t = j / sample_rate;

        // Generate sine wave
        float signal = amplitude * std::cos(2.0f * M_PI * frequency * t);

        // Add noise
        float noise = (std::rand() / float(RAND_MAX) - 0.5f) * noise_level;

        // Store as complex number (real part only for simplicity)
        data[{0, j}] = CF32(signal + noise, 0.0f);
    }
}

/**
 * Jetstream Superluminal Interface Demo
 *
 * This example demonstrates the key capabilities of Superluminal:
 * - Text and Markdown rendering
 * - Image display
 * - Interactive controls (sliders, buttons)
 * - Real-time signal plotting with domain transforms
 */
int main() {
    std::cout << "Initializing Superluminal Interface..." << std::endl;

    // Create memory buffer
    auto data = Tensor<Device::CPU, CF32>({1, 8192});

    // Create initial test signal
    GenerateTestSignal(data);

    // Declare variables
    float frequency = 1000.0f;
    float amplitude = 1.0f;
    float noise_level = 0.1f;

    // Interactive controls section
    Superluminal::Box("Controls", {{1, 0, 0}, {1, 0, 0}}, [&]{
        // Markdown documentation example

        Superluminal::Markdown(
            "# Key Features\n"
            "- **Signal Generation**: Synthetic sine waves with noise.\n"
            "- **Interactive Controls**: Real-time parameter adjustment.\n"
            "- **Visualization**: Time to frequency domain transforms.\n"
        );

        // Interactive sliders for signal parameters

        Superluminal::Markdown("# Signal Parameters");
        Superluminal::Slider("Frequency (Hz)", 100.0f, 5000.0f, frequency);
        Superluminal::Slider("Amplitude", 0.1f, 2.0f, amplitude);
        Superluminal::Slider("Noise Level", 0.0f, 0.5f, noise_level);
    });

    // Signal visualization - shows domain transform capability
    Superluminal::Plot("Signal Visualization", {{0, 1, 1}, {0, 1, 1}}, {
        .buffer = data,
        .type = Superluminal::Type::Line,
        .source = Superluminal::Domain::Time,       // Input is time domain data
        .display = Superluminal::Domain::Frequency, // Display as frequency domain
    });

    Superluminal::RealtimeLoop([&](const bool& running){
        while (running) {
            GenerateTestSignal(data, frequency, amplitude, noise_level);

            Superluminal::Update();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    std::cout << "Superluminal demo completed successfully!" << std::endl;

    return 0;
}
