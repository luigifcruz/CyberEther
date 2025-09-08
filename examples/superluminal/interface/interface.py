from dataclasses import dataclass
import superluminal as lm
import numpy as np
import time
import math

print("Initializing Superluminal Interface...")

# Create a data class to hold interactive state
@dataclass
class SignalState:
    frequency = [1000.0]
    amplitude = [1.0]
    noise_level = [0.1]

def generate_test_signal(data, frequency=1000.0, amplitude=1.0, noise_level=0.1):
    """
    Generates a test signal combining a sine wave with noise
    This demonstrates typical signal processing scenarios
    """
    sample_rate = 44100.0

    for j in range(data.shape[1]):
        t = j / sample_rate

        # Generate sine wave (using cosine to match C++ version)
        signal = amplitude * math.cos(2.0 * math.pi * frequency * t)

        # Add noise
        noise = (np.random.rand() - 0.5) * noise_level

        # Store as complex number (real part only for simplicity)
        data[0, j] = complex(signal + noise, 0.0)

# Initialize state
state = SignalState()

# Create memory buffer
data = np.zeros((1, 8192), dtype=np.complex64)

# Generate initial test signal
generate_test_signal(data)

# Interactive controls section
lm.box("Controls", [[1, 0, 0], [1, 0, 0]], lambda: [
    # Markdown documentation example
    lm.markdown(
        "# Key Features\n"
        "- **Signal Generation**: Synthetic sine waves with noise.\n"
        "- **Interactive Controls**: Real-time parameter adjustment.\n"
        "- **Visualization**: Time to frequency domain transforms.\n"
    ),

    # Interactive sliders for signal parameters
    lm.markdown("# Signal Parameters"),
    lm.slider("Frequency (Hz)", 100.0, 5000.0, state.frequency),
    lm.slider("Amplitude", 0.1, 2.0, state.amplitude),
    lm.slider("Noise Level", 0.0, 0.5, state.noise_level),
])

# Signal visualization
lm.plot(data, lm.line, mosaic=[[0, 1, 1], [0, 1, 1]], label="Sine")

def realtime_callback():
    """
    Jetstream Superluminal Interface Demo

    This example demonstrates the key capabilities of Superluminal:
    - Text and Markdown rendering
    - Image display
    - Interactive controls (sliders, buttons)
    - Real-time signal plotting with domain transforms
    """
    while lm.running():
        # Generate updated signal with current parameters
        generate_test_signal(
            data,
            frequency=state.frequency[0],
            amplitude=state.amplitude[0],
            noise_level=state.noise_level[0]
        )

        # Update the interface
        lm.update()

        # Sleep for 10 ms to maintain smooth updates
        time.sleep(0.01)

# Run the realtime loop
lm.realtime(realtime_callback)

print("Superluminal demo completed successfully!")
