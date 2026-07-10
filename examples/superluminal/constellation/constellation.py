import numpy as np
import superluminal as lm

points = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64) / np.sqrt(2)
noise = (np.random.randn(1, 2048) + 1j * np.random.randn(1, 2048)) * 0.05
data = (np.random.choice(points, size=(1, 2048)) + noise).astype(np.complex64)
lm.plot(data, lm.scatter, label="QPSK")

lm.show()
