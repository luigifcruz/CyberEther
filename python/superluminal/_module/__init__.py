import time
import threading
import numpy as np
import os
import tempfile
import urllib.request

import superluminal._internal as lm

try:
    import cupy as cp
except:
    cp = None

_url_cache = {}

def plot(data: np.ndarray,
         type: lm.constant,
         label: str = "",
         mosaic: list[list[int]] = [[1]],
         domain: tuple[lm.constant, lm.constant] = (lm.time, lm.time),
         operation: lm.constant = lm.amplitude,
         batch_axis: int = -1,
         channel_axis: int = -1,
         channel_index: int = -1,
         options: dict[str, any] = {}):
    #
    # Check classes.
    #

    # Check data.

    if not isinstance(data, np.ndarray) and (cp and not isinstance(data, cp.ndarray)):
        raise TypeError("Data must be a numpy or cupy array.")

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
    # Check batch.
    #

    # TODO: Implement.

    #
    # Check channel.
    #

    if (channel_axis + 1) > len(data.shape):
        raise ValueError(f"Invalid channel axis: {channel_axis}")

    if (channel_index + 1) > data.shape[channel_axis]:
        raise ValueError(f"Invalid channel index: {channel_index}")

    #
    # Plot.
    #

    cfg = lm.plot_config()
    cfg.buffer = data
    cfg.type = type.value
    cfg.source = domain[0].value
    cfg.display = domain[1].value
    cfg.operation = operation.value
    cfg.batch_axis = batch_axis
    cfg.channel_axis = channel_axis
    cfg.channel_index = channel_index
    cfg.options = options

    lm.plot(label, mosaic, cfg)

def configure(preferred_device: lm.constant = lm.cpu,
              device_id: int = 0,
              window_title: str = "Superluminal"):
    cfg = lm.instance_config()
    cfg.device_id = device_id
    cfg.preferred_device = preferred_device.value
    cfg.window_title = window_title
    lm.initialize(cfg)

def show():
    lm.start()
    lm.update()

    while lm.presenting():
        lm.poll_events(False)
        time.sleep(0.01)

    lm.stop()
    lm.terminate()

def running():
    return lm.presenting()

def realtime(callback):
    lm.start()

    t = threading.Thread(target=callback)
    t.start()

    while lm.presenting():
        lm.poll_events(False)
        time.sleep(0.02)

    t.join()

    lm.stop()
    lm.terminate()

def layout(matrix_height, matrix_width,
           panel_height, panel_width,
           offset_x, offset_y):
    return lm.mosaic_layout(matrix_height, matrix_width,
                            panel_height, panel_width,
                            offset_x, offset_y)

def box(title: str, mosaic: list[list[int]], callback):
    """
    Create a box with UI elements.

    Args:
        title: The title of the box
        mosaic: The layout mosaic (e.g., [[1]] for full screen)
        callback: A function that defines the UI elements
    """
    if not isinstance(title, str):
        raise TypeError("Title must be a string.")

    if not isinstance(mosaic, list):
        raise TypeError("Mosaic must be a list.")

    if not callable(callback):
        raise TypeError("Callback must be callable.")

    lm.box(title, mosaic, callback)

def text(format_string: str, *args):
    """
    Display formatted text.

    Args:
        format_string: The format string (can include {} placeholders)
        *args: Arguments to format into the string
    """
    if args:
        content = format_string.format(*args)
    else:
        content = format_string
    lm.text(content)

def slider(label: str, min_val: float, max_val: float, value: list):
    """
    Display a slider control.

    Args:
        label: The label for the slider
        min_val: Minimum value
        max_val: Maximum value
        value: A list containing a single value (modified in place)
    """
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("Value must be a list with exactly one element")

    lm.slider(label, min_val, max_val, value)

def markdown(content: str):
    """
    Display markdown formatted text.

    Args:
        content: The markdown content to display
    """
    if not isinstance(content, str):
        raise TypeError("Content must be a string.")

    lm.markdown(content)

def image(filepath: str, width: float = -1.0, height: float = -1.0, fit_to_window: bool = False):
    """
    Display an image from a file or URL.

    Args:
        filepath: Path to the local image file or a URL.
        width: Display width in pixels. If -1, auto-calculate from height or use original.
        height: Display height in pixels. If -1, auto-calculate from width or use original.
        fit_to_window: If True, scale image to fit available window space while preserving aspect ratio.
    """

    # Handle URL
    if filepath.startswith(('http://', 'https://')):
        if filepath in _url_cache and os.path.exists(_url_cache[filepath]):
            filepath = _url_cache[filepath]
        else:
            request = urllib.request.Request(filepath, headers={'User-Agent': 'Superluminal/1.0'})
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filepath)[1]) as tmp:
                with urllib.request.urlopen(request) as response:
                    tmp.write(response.read())
                _url_cache[filepath] = tmp.name
                filepath = tmp.name

    # Handle relative path
    elif not os.path.isabs(filepath):
        filepath = os.path.join(os.environ.get("PWD", os.getcwd()), filepath)

    lm.image(filepath, float(width), float(height), fit_to_window)
