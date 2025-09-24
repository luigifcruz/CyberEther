# AI

## 1. Prompt for generating a Block description.

```
Write the description of the block structuring it in this way:
1. Summary: Extended version of the summary but still simple and easy to understand. Should crearly describe what the block does in a few phrases. Should be titled "## Summary:"
2. Arguments: A bullet point list describing each argument. Should be titled "## Arguments:". Each argument should be in bold e.g. "- **Axis**: ..."
3. Useful For: A bullet point list describer very good uses for this block. Should be titled "## Useful For:"
4. Examples: Examples of input and output. Should talk in terms of block configuration and input and output shape and type. Should be titled "## Examples:"
5. Implementation: Should clearly tell the user how this module works internally. Should be titled "## Implementation:"

Ground Rules:
- Don't forget to add a period to every phrase.
- Refering to a tensor shape and type should be done with the following syntax: `CF32[8192, 128]`.

Example:
std::string description() const {
    return "The Decimator block reduces tensor dimensionality by summing elements along a specified axis, "
            "then extracting the first element of the reduced dimension. This effectively collapses "
            "multi-dimensional data into lower-dimensional output while preserving accumulated values.\n\n"

            "## Parameters\n"
            "- **Axis**: The axis along which to sum and slice the input tensor.\n\n"

            "## Useful For:\n"
            "- Implementing simple decimation filters for signal processing.\n"
            "- Aggregating sensor data from multiple sources.\n\n"

            "## Examples:\n"
            "- Time-domain decimation:\n"
            "  Config: Axis=1\n"
            "  Input: CF32[8192, 128] → Output: CF32[8192]\n\n"

            "## Implementation:\n"
            "Input → Add Axis → Squeeze Axis → Duplicate → Output\n"
            "1. Arithmetic module sums all elements along the specified axis.\n"
            "2. Tensor modifier slices the result to extract index 0 from the reduced dimension.\n"
            "3. Duplicate module ensures proper output buffering and host accessibility.";
}
```

## 2. To create a new module or block:

```
...

Make sure to follow the coding style of the existing blocks and modules. For the block configuration parameters use patterns from existing blocks but feel free to implement something new if necessary. You can find information how to add them via the CONTRIBUTING.md documentation. Implement only for the CPU device and ignore the rest. Implement only CF32 and F32 and leave out the other types. Do not create benchmark.

[[#1]]
```

## 3. Project validation:

```
Rules:
1. Do not create any comment in the code.
2. Do not create includes.
3. Do not fix errors not listed in the validation steps. Just report them in the end.

Project validation steps:
1. Make sure all internal modules of each block are destroyed in reverse order of creation.
2. Make sure all internal modules of each block are guarded with if (...) {} before destruction.
3. Make sure the all output tensors of each module (located inside struct Output {...}) are allocated only if the module create is guaranteed to work (without JST_CHECK(...)).
```


There is a big bottleneck on the Vulkan render backend that is that every time i need to update the buffer, the Backend::ExecuteOnce is called. This probably stalls the whole pipeline. I would like to apply a trick that i used on the kernel.cc that is to schedule an update for the next time the render is encoding the present(). Make sure that partial buffer updates are all batched (dont discard the last one that was not applied yet if the user calls update multiple times inbetween presents). The bind() method should register the buffer in the window state and the buffer should be encoded by the window before calling the surface encoding.
