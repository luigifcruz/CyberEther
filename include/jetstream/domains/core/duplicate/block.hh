#ifndef JETSTREAM_DOMAINS_CORE_DUPLICATE_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_DUPLICATE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Duplicate : public Block::Config {
    std::string outputDevice = "cpu";
    bool hostAccessible = true;

    JST_BLOCK_TYPE(duplicate);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(hostAccessible, outputDevice);
    JST_BLOCK_DESCRIPTION(
        "Duplicate",
        "Copies and transfers signal data.",
        "# Duplicate\n"
        "\n"
        "Copies each input buffer into a new buffer, on the same device or another one."
        " It usually follows a Slice with Contiguous off, or sits where a chain moves "
        "between CPU and GPU memory.\n"
        "\n"
        "- **Output Device defaults to CPU.** A GPU input stays on its device only with"
        " Output Device set to None.\n"
        "- **Strided input comes out contiguous.** Blocks that reject strided buffers, "
        "such as Reshape, can follow it.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape. |\n"
        "| **Output** (out) | Same as the input | Same shape, with every axis and "
        "attribute kept, in a contiguous buffer on Output Device. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Output Device** | CPU | None, CPU, CUDA, Metal, Vulkan | Memory the output"
        " lives in. None keeps the device of the input. |\n"
        "| **Host Accessible** | On | On, Off | Makes the output readable from the CPU."
        " Hidden for CPU. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Make a slice contiguous | Defaults | 1,024 samples from a Slice with "
        "Contiguous off | 1,024 samples in a new CPU buffer |\n"
        "| Send samples to the GPU | Output Device CUDA | 8 batches x 8,192 samples in "
        "CPU memory | 8 batches x 8,192 samples in CUDA memory the CPU can read |\n"
        "| Bring results back | Defaults | 8,192 bins in CUDA memory | 8,192 bins in "
        "CPU memory |\n"
        "| Copy on the same device | Output Device None | 8,192 samples in CUDA memory "
        "| 8,192 samples in a new CUDA buffer |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the output device has no buffer backend in this build"
        " | Output Device names a backend the build lacks, such as CUDA on macOS | Pick"
        " CPU, or None to stay on the input device. |\n"
        "| The block reports that the output must be host accessible for a CPU input | "
        "Host Accessible is off for a CUDA output, or a Vulkan output on a GPU with its"
        " own memory | Turn on Host Accessible. |\n"
        "| The block reports that it cannot map a Metal output to a CUDA input | A CUDA"
        " input can only be copied to CPU, CUDA, or Vulkan | Pick CPU, CUDA, or None. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each cycle copies the whole input in one transfer when it is contiguous, and "
        "element by element in order when it is strided. When the devices differ, the "
        "copy writes straight into the output memory as the input device sees it, so "
        "some outputs must be host accessible.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Slice** with Contiguous on makes its own contiguous copy.\n"
        "- **Cast** converts the type instead of the device."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_DUPLICATE_BLOCK_HH
