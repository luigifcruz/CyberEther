import time
import numpy as np
import superluminal as lm

print("Welcome to Superluminal!")

data = np.random.rand(1, 8192).astype(np.complex64)

lm.plot(data, lm.line, label="Sine")

def callback():
    phase = 0

    while lm.running():
        # Generate sine wave
        for i in range(data.shape[1]):
            data[0, i] = np.sin(i * 0.01 + phase) * 20 + 30
        
        phase += 0.1

        # Update the plot.
        lm.update()

        # Sleep for 10 ms.
        time.sleep(0.01)

lm.realtime(callback)

print("Goodbye from Superluminal!")