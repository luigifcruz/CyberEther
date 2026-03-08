import numpy as np

import superluminal as lm

data = np.random.rand(1, 8192).astype(np.float32)

lm.configure(remote=True)
lm.print_remote_info()

lm.plot(data, lm.line, label="Remote Demo", domain=(lm.time, lm.frequency))
lm.show()
