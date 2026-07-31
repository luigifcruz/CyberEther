#ifndef JETSTREAM_MEMORY_AXIS_HH
#define JETSTREAM_MEMORY_AXIS_HH

#include <optional>
#include <string_view>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/memory/types.hh"

namespace Jetstream {

class Tensor;

inline constexpr std::string_view SampleAxisAttribute = "sampleAxis";
inline constexpr std::string_view BatchAxisAttribute = "batchAxis";
inline constexpr std::string_view ChannelAxisAttribute = "channelAxis";

struct SignalAxes {
    std::optional<Index> sample;
    std::optional<Index> batch;
    std::optional<Index> channel;
};

using AxisMap = std::vector<std::optional<Index>>;

JETSTREAM_API std::optional<Index> ResolveAxis(I64 axis, Index rank);
JETSTREAM_API std::optional<Index> ResolveInsertionAxis(I64 axis, Index rank);

JETSTREAM_API Result ResolveSignalAxes(const Tensor& tensor, SignalAxes& axes);
JETSTREAM_API Result SetSignalAxes(Tensor& tensor, const SignalAxes& axes);
JETSTREAM_API Result MapSignalAxes(const Tensor& tensor,
                                   const AxisMap& axisMap,
                                   SignalAxes& axes);

}  // namespace Jetstream

#endif  // JETSTREAM_MEMORY_AXIS_HH
