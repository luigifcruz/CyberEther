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
    template<Device DeviceId, typename Type>
    Result linkOutput(const std::string& name,
                      Tensor<DeviceId, Type>& dst,
                      const Tensor<DeviceId, Type>& src) {
        Result res = Result::SUCCESS;

        if (!dst.empty()) {
            JST_ERROR("The destination buffer should be empty during initialization.");
            res |= Result::ERROR;
        }

        if (src.empty()) {
            JST_ERROR("The source buffer shouldn't be empty during initialization.");
            res |= Result::ERROR;
        }

        dst = src;

        // Rename the pinId to the bundle's name.
        const auto& srcLocale = src.template store<Locale>("locale");
        dst.store()["locale"] = Locale{srcLocale.id, srcLocale.subId, name};

        return res;
    }
};

}  // namespace Jetstream

#endif
