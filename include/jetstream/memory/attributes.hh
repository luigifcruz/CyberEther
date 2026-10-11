#ifndef JETSTREAM_MEMORY_ATTRIBUTES_HH
#define JETSTREAM_MEMORY_ATTRIBUTES_HH

#include <any>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/memory/types.hh"

namespace Jetstream {

class Tensor;

inline constexpr std::string_view FrequencyAttribute = "frequency";
inline constexpr std::string_view SampleRateAttribute = "sampleRate";
inline constexpr std::string_view BandwidthAttribute = "bandwidth";
inline constexpr std::string_view CenterAttribute = "center";
inline constexpr std::string_view SampleAxisAttribute = "sampleAxis";
inline constexpr std::string_view BatchAxisAttribute = "batchAxis";
inline constexpr std::string_view ChannelAxisAttribute = "channelAxis";
inline constexpr std::string_view ChannelOffsetsAttribute = "channelOffsets";
inline constexpr std::string_view ChannelPhaseIncrementsAttribute = "channelPhaseIncrements";

JETSTREAM_API Result ValidateAttribute(std::string_view key, const std::any& value);
JETSTREAM_API Result AttributeTypeError(std::string_view key,
                                        std::string_view expected,
                                        const std::any& value);

JETSTREAM_API Result ReadAttribute(const Tensor& tensor,
                                   const std::string& key,
                                   std::any& value);
JETSTREAM_API Result ReadAttribute(const Tensor& tensor,
                                   const std::string& key,
                                   std::optional<F32>& value);
JETSTREAM_API Result ReadAttribute(const Tensor& tensor,
                                   const std::string& key,
                                   std::optional<U64>& value);
JETSTREAM_API Result ReadAttribute(const Tensor& tensor,
                                   const std::string& key,
                                   std::optional<std::vector<U64>>& value);
JETSTREAM_API Result ReadAttribute(const Tensor& tensor,
                                   const std::string& key,
                                   std::optional<std::vector<F64>>& value);

}  // namespace Jetstream

#endif  // JETSTREAM_MEMORY_ATTRIBUTES_HH
