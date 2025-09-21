import time
import numpy as np
import superluminal as lm

# Configuration constants
M_HEIGHT = 6
M_WIDTH = 11
DATA_SHAPE = (42, 8192)
RFI_CHANCE = 0.01
SETI_CHANCE = 0.01
SIGNAL_DEACTIVATION_TIME = 1.5
UPDATE_INTERVAL = 0.001
NOISE_STD = np.sqrt((1-0)**2/12)

# Initialize shared data buffer
data = np.random.rand(*DATA_SHAPE).astype(np.complex64)


def should_skip_plot_position(x, y):
    """Determine if a plot position should be skipped based on layout rules."""
    if y == 0 or y == 5:
        return x < 3 or x > 7
    elif y == 2 or y == 3:
        return x < 1 or x > 9
    else:
        return x < 2 or x > 8


def create_plots():
    """Create all waterfall plots in the mosaic layout."""
    plot_index = 0
    for y, x in np.ndindex((M_HEIGHT, M_WIDTH)):
        if should_skip_plot_position(x, y):
            continue

        lm.plot(data=data,
                type=lm.waterfall,
                channel_axis=0,
                channel_index=plot_index,
                label=f"plot{plot_index}",
                mosaic=lm.layout(M_HEIGHT, M_WIDTH, 1, 1, x, y))
        plot_index += 1


def create_annotations():
    """Create annotations for the mosaic layout."""
    lm.box("Annotations", lm.layout(M_HEIGHT, M_WIDTH, 1, 3, 0, 0), lambda: [
        lm.markdown(
            "# Interferometer Simulation\n"
            "Simulates 42 radio telescope antennas collecting data in real-time. "
            "Watch for SETI signals drifting across frequency bands and random "
            "radio interference bursts in the waterfall displays."
        ),
    ])


def generate_base_data():
    """Generate base noise data with optional RFI injection."""
    new_data = np.random.rand(*DATA_SHAPE).astype(np.complex64)

    # Random chance to inject RFI
    if np.random.rand() < RFI_CHANCE:
        new_data += NOISE_STD * 50 * np.random.rand(*DATA_SHAPE)

    return new_data


def create_seti_signal():
    """Create a new SETI signal with random parameters."""
    return {
        'element': np.random.randint(DATA_SHAPE[0]),
        'index': np.random.rand() * DATA_SHAPE[1],
        'slope': 32.0 + 64.0 * np.random.rand(),
        'magnitude': NOISE_STD * (50 + np.random.randint(50))
    }


def inject_seti_signals(new_data, seti_signals):
    """Inject SETI signals into the data and update signal positions."""
    active_signals = []

    for signal in seti_signals:
        element = signal['element']
        start_index = int(signal['index'])
        magnitude = signal['magnitude']

        # Calculate signal width and bounds
        signal_width = 128 + int(256.0 * np.random.rand())
        end_index = min(start_index + signal_width, DATA_SHAPE[1])

        # Inject signal into data
        new_data[element, start_index:end_index] += magnitude

        # Update signal position
        signal['index'] += signal['slope']

        # Keep signal if still within bounds
        if signal['index'] < DATA_SHAPE[1]:
            active_signals.append(signal)

    return active_signals


def update_signal_tracking(seti_signals, last_seen):
    """Update tracking of active signals and deactivate old ones."""
    current_time = time.time()

    # Track currently active elements
    for signal in seti_signals:
        last_seen[signal['element']] = current_time

    # Deactivate plots for elements not seen recently
    for element in list(last_seen.keys()):
        if current_time - last_seen[element] > SIGNAL_DEACTIVATION_TIME:
            del last_seen[element]


def simulation_callback():
    """Main simulation loop that generates and updates data."""
    iframe = 0
    seti_signals = []
    last_seen = {}

    while lm.running():
        # Generate new base data
        new_data = generate_base_data()

        # Chance to create new SETI signal
        if np.random.rand() < SETI_CHANCE:
            seti_signals.append(create_seti_signal())

        # Inject existing SETI signals
        seti_signals = inject_seti_signals(new_data, seti_signals)

        # Update signal tracking
        update_signal_tracking(seti_signals, last_seen)

        # Update shared data buffer
        np.copyto(data, new_data)

        # Refresh visualization
        lm.update()

        # Sleep for update interval
        time.sleep(UPDATE_INTERVAL)
        iframe += 1


# Initialize plots and start simulation
create_annotations()
create_plots()

lm.realtime(simulation_callback)
