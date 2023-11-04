#ifndef JETSTREAM_MODULE_HH
#define JETSTREAM_MODULE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/metadata.hh"
#include "jetstream/interface.hh"
#include "jetstream/render/base.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

class JETSTREAM_API Module : public Interface {
 public:
    virtual ~Module() = default;

    virtual Result create() = 0;
    // TODO: Validate modules lifetime.
    virtual Result destroy() {
        return Result::SUCCESS;
    }

    virtual void summary() const = 0;

 protected:
    template<Device DeviceId, typename DataType>
    Result initInput(const std::string&, Tensor<DeviceId, DataType>& buffer) {
        if (buffer.empty()) {
            JST_ERROR("Input is empty during initialization.");
            return Result::ERROR;
        }
        return Result::SUCCESS;
    }

    template<Device DeviceId, typename DataType>
    Result initOutput(const std::string& name,
                      Tensor<DeviceId, DataType>& buffer,
                      const std::vector<U64>& shape,
                      const Result& prevRes) {
        Result res = Result::SUCCESS;

        if (!buffer.empty()) {
            JST_ERROR("The output buffer should be empty during initialization.");
            res |= Result::ERROR;
        }

        if (prevRes == Result::SUCCESS) {
            buffer = Tensor<DeviceId, DataType>(shape);
        }

        buffer.store()["locale"] = Locale{locale().id, locale().subId, name};

        return res;
    }

    template<Device DeviceId, typename Type>
    Result initInplaceOutput(const std::string& name,
                             Tensor<DeviceId, Type>& dst,
                             Tensor<DeviceId, Type>& src,
                             const Result& prevRes = Result::SUCCESS) {
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
        dst.updateLocale({locale().id, locale().subId, name});

        return res;
    }

    template<Device DeviceId, typename DataType>
    Result voidOutput(Tensor<DeviceId, DataType>& buffer) {
        const auto vectorName = buffer.template store<Locale>("locale").pinId;
        buffer = Tensor<DeviceId, DataType>();
        buffer.store()["locale"] = Locale{locale().id, locale().subId, vectorName};

        return Result::SUCCESS;
    }

    // TODO: Add InitInplaceOutput with reshape.
};

class JETSTREAM_API Compute {
 public:
    virtual ~Compute() = default;

    virtual constexpr Result createCompute(const RuntimeMetadata& meta) = 0;
    virtual constexpr Result compute(const RuntimeMetadata& meta) = 0;
    virtual constexpr Result computeReady() {
        return Result::SUCCESS;
    }
    virtual constexpr Result destroyCompute(const RuntimeMetadata&) {
        return Result::SUCCESS;
    }

 protected:
    friend Instance;
};

class JETSTREAM_API Present {
 public:
    virtual ~Present() = default;

    virtual constexpr Result createPresent() = 0;
    virtual constexpr Result present() = 0;
    virtual constexpr Result destroyPresent() = 0;

 protected:
    std::shared_ptr<Render::Window> window;

    friend Instance;
};

}  // namespace Jetstream

#endif
