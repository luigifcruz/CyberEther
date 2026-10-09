#ifndef JETSTREAM_DOMAINS_CORE_RANGE_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_RANGE_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Range : public Block::Config {
    F32 min = -1.0f;
    F32 max = +1.0f;

    JST_BLOCK_TYPE(range);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(min, max);
    JST_BLOCK_DESCRIPTION(
        "Range",
        "Normalizes input using an affine range mapping.",
        "# Range\n"
        "\n"
        "Rescales values along a straight line so that Min lands on 0 and Max lands on "
        "1. It usually follows an Amplitude block that gives levels in dB, and feeds a "
        "Waterfall that colors values from 0 to 1.\n"
        "\n"
        "- **Values outside Min and Max are not clipped.** They land below 0 or above "
        "1, so averages downstream stay exact.\n"
        "- **The defaults suit values from -1 to 1.** Set Min and Max around the signal"
        " level for input in dB.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 | Any shape. Strided input is accepted. |\n"
        "| **Output** (out) | F32 | Same shape, with every axis and attribute kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Min** | -1 dBFS | Any, slider -300 to 0 dBFS | Input value mapped to 0. "
        "The lower of Min and Max always maps to 0. |\n"
        "| **Max** | 1 dBFS | Any, slider -300 to 0 dBFS | Input value mapped to 1. "
        "Equal to Min, every output is 0.5. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Spectrum to waterfall colors | Min -100, Max 0 | 8,192 bins from -100 to 0 "
        "dBFS | 8,192 values from 0 to 1 |\n"
        "| Focus on strong signals | Min -80, Max -40 | 8,192 bins from -100 to -30 "
        "dBFS | 8,192 values from -0.5 to 1.25 |\n"
        "| Audio samples to 0 to 1 | Defaults | 1,024 samples from -1 to 1 | 1,024 "
        "values from 0 to 1 |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| A Waterfall downstream shows one flat color | Min and Max are at their "
        "defaults of -1 and 1 while the input is in dB, so a level of -60 dBFS maps to "
        "-29.5 | Set Min and Max around the signal, such as -100 and 0. |\n"
        "| The block reports an unsupported data type | The input is not F32, such as "
        "complex samples | Convert complex samples to dB with an Amplitude block first."
        " |\n"
        "| Values fall below 0 or above 1 | The input reaches past Min or Max, and the "
        "block does not clip | Widen Min and Max to cover the signal. |\n"
        "| Every output value is 0.5 | Min equals Max | Move them apart. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block applies one scale and offset to every value, after putting the "
        "bounds in order. Infinite results are pinned to the nearest end, and NaN goes "
        "to the low end. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def normalize(x, min_=-1.0, max_=1.0):\n"
        "    lo, hi = sorted((min_, max_))\n"
        "    if lo == hi:\n"
        "        return np.full_like(x, 0.5)\n"
        "    scale = 1.0 / (hi - lo)\n"
        "    y = x * scale - lo * scale\n"
        "    return np.where(np.isfinite(y), y, np.where(y > 0, 1.0, 0.0))\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Amplitude** turns samples into the dB levels it rescales.\n"
        "- **Spectrum Engine** rescales its own output with Enable Scale.\n"
        "- **Waterfall** colors values from 0 to 1."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_RANGE_BLOCK_HH
