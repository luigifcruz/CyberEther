import numpy as np
import superluminal as lm

print("Welcome to Superluminal!")

data = np.random.randint(0, 3, size=(1, 8192)).astype(np.float32)
lm.plot(data, lm.line, label="Random", domain=(lm.time, lm.frequency))

lm.show()

print("Goodbye from Superluminal!")
