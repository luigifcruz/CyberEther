import numpy as np
import superluminal as lm

print("Welcome to Superluminal!")

data = np.random.rand(1, 8192).astype(np.complex64)

lm.plot(data, lm.line, label="Random", domain=(lm.time, lm.frequency))
lm.show()

print("Goodbye from Superluminal!")