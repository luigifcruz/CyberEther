#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_BLADERF_QUICK_TUNE_BUDGET_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_BLADERF_QUICK_TUNE_BUDGET_HH

#include <optional>
#include <vector>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

class JETSTREAM_API QuickTuneBudget {
 public:
    enum class Decision : U8 {
        Reuse,
        Build,
        Fallback,
    };

    explicit QuickTuneBudget(std::optional<U64> capacity = std::nullopt);

    Decision plan(const std::vector<F64>& frequencies) const;
    void built(const std::vector<F64>& frequencies);
    void failed(U64 allocated);
    U64 used() const;
    std::optional<U64> remaining() const;

 private:
    std::optional<U64> capacity;
    U64 allocated = 0;
    std::vector<F64> prepared;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_BLADERF_QUICK_TUNE_BUDGET_HH
