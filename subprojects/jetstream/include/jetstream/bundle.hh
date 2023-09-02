#ifndef JETSTREAM_BUNDLE_HH
#define JETSTREAM_BUNDLE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/interface.hh"

namespace Jetstream {

class JETSTREAM_API Bundle : public Interface {
 public:
    virtual ~Bundle() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;

 protected:
    template<Device DeviceId, typename Type, U64 Dimensions>
    Result linkOutput(const std::string& name,
                      Vector<DeviceId, Type, Dimensions>& dst,
                      const Vector<DeviceId, Type, Dimensions>& src) {
        if (!dst.empty()) {
            JST_ERROR("[BUNDLE] The destination buffer should be empty during initialization.");
            return Result::ERROR;
        }

        if (src.empty()) {
            JST_ERROR("[BUNDLE] The source buffer shouldn't be empty during initialization.");
            return Result::ERROR;
        }

        dst = src;

        // Rename the pinId to the bundle's name.
        const auto& locale = src.locale();
        dst.updateLocale({locale.id, locale.subId, name});

        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif
