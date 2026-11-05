#include "ownership.hh"

#include <mutex>
#include <set>
#include <utility>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {
struct OwnershipRegistry {
    std::mutex mutex;
    std::set<std::string> devices;
};

OwnershipRegistry& Registry() {
    // Keep the registry alive for owners released during static destruction.
    // The static pointer is trivially destructible; the registry lives until exit.
    static auto* registry = new OwnershipRegistry;
    return *registry;
}
}  // namespace

BladeRfOwnership::~BladeRfOwnership() {
    if (!serial.empty()) {
        auto& registry = Registry();
        std::lock_guard lock(registry.mutex);
        registry.devices.erase(serial);
    }
}

Result BladeRfOwnership::acquire(const std::string& requested) {
    auto& registry = Registry();
    std::lock_guard lock(registry.mutex);
    if (!serial.empty() || requested.empty()) {
        JST_ERROR(
            "[SDR_BLADERF] Cannot acquire ownership without a serial "
            "or twice on one handle.");
        return Result::ERROR;
    }
    std::string acquired = requested;
    if (!registry.devices.insert(acquired).second) {
        JST_ERROR(
            "[SDR_BLADERF] Device '{}' is already in use. "
            "Use one SDR block with Channels set to 2 for both receivers.",
            requested);
        return Result::ERROR;
    }
    serial = std::move(acquired);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
