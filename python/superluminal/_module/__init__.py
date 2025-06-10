import time
import threading
import numpy as np

import superluminal._internal as lm

def plot(data: np.ndarray,
         type: lm.constant,
         label: str = "",
         mosaic: list[list[int]] = [[1]],
         domain: tuple[lm.constant, lm.constant] = (lm.time, lm.time),
         operation: lm.constant = lm.amplitude,
         options: dict[str, any] = {}):
    #
    # Check classes.
    #

    # Check data.

    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")

    # Check type.

    if not isinstance(type, lm.constant):
        raise TypeError("Type must be a constant.")

    # Check label.

    if not isinstance(label, str):
        raise TypeError("Label must be a string.")

    # Check mosaic.

    if not isinstance(mosaic, list):
        raise TypeError("Mosaic must be a list.")

    if len(mosaic) < 1:
        raise ValueError("Mosaic must be a list of at least one list.")

    for row in mosaic:
        if not isinstance(row, list):
            raise TypeError("Mosaic must be a list of lists.")

        if len(row) < 1:
            raise ValueError("Mosaic must be a list of lists of at least one element.")

        for element in row:
            if not isinstance(element, int):
                raise TypeError("Mosaic must be a list of lists of integers.")

    # Check domain.

    if not isinstance(domain, tuple):
        raise TypeError("Domain must be a tuple of two constants.")

    if len(domain) != 2:
        raise ValueError("Domain must be a tuple of two constants.")

    # Check operation.

    if not isinstance(operation, lm.constant):
        raise TypeError("Operation must be a constant.")

    # Check options.

    if not isinstance(options, dict):
        raise TypeError("Options must be a dictionary.")

    for k, _ in options.items():
        if not isinstance(k, str):
            raise TypeError("Options keys must be strings.")

    #
    # Check constants.
    #

    if type.key not in lm._types_lst:
        raise ValueError(f"Invalid type: {type.key}")

    if domain[0].key not in lm._domains_lst:
        raise ValueError(f"Invalid source domain: {domain[0].key}")

    if domain[1].key not in lm._domains_lst:
        raise ValueError(f"Invalid display domain: {domain[1].key}")

    if operation.key not in lm._operations_lst:
        raise ValueError(f"Invalid operation: {operation.key}")

    #
    # Plot.
    #

    cfg = lm.plot_config()
    cfg.buffer = data
    cfg.type = type.value
    cfg.source = domain[0].value
    cfg.display = domain[1].value
    cfg.operation = operation.value
    cfg.options = options

    lm.plot(label, mosaic, cfg)

def show():
    lm.start()
    lm.update()
    lm.block()
    lm.stop()
    lm.terminate()

def running():
    return lm.presenting()

def realtime(callback):
    lm.start()

    threading.Thread(target=callback).start()

    while lm.presenting():
        lm.poll_events(False)
        time.sleep(0.02)

    lm.stop()
    lm.terminate()