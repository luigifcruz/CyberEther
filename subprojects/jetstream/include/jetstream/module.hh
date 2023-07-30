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

    virtual void summary() const = 0;

 protected:
    template<Device DeviceId, typename DataType, U64 Dimensions>
    Result initInput(const Vector<DeviceId, DataType, Dimensions>& buffer) {
        if (buffer.empty()) {
            JST_FATAL("Module input can't be empty.");
            return Result::ERROR;
        }
        return Result::SUCCESS;
    }

    template<Device DeviceId, typename DataType, U64 Dimensions>
    Result initOutput(Vector<DeviceId, DataType, Dimensions>& buffer,
                      const VectorShape<Dimensions>& shape) {
        if (!buffer.empty()) {
            JST_FATAL("The output buffer should be empty on initialization.");
            return Result::ERROR;
        }

        buffer = Vector<DeviceId, DataType, Dimensions>(shape);

        return Result::SUCCESS;
    }

    template<Device DeviceId, typename Type, U64 Dimensions>
    Result initInplaceOutput(Vector<DeviceId, Type, Dimensions>& dst,
                             const Vector<DeviceId, Type, Dimensions>& src) {
        dst = const_cast<Vector<DeviceId, Type, Dimensions>&>(src);
        dst.increasePosCount();

        return Result::SUCCESS;
    }

    // TODO: Add initInplaceOutput with reshape.
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
};

class JETSTREAM_API Present {
 public:
    virtual ~Present() = default;

    virtual constexpr Result createPresent(Render::Window& window) = 0;
    virtual constexpr Result present(Render::Window& window) = 0;
    virtual constexpr Result destroyPresent(Render::Window&) {
        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif
