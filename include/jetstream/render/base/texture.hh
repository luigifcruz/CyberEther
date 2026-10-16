#ifndef JETSTREAM_RENDER_BASE_TEXTURE_HH
#define JETSTREAM_RENDER_BASE_TEXTURE_HH

#include <atomic>
#include <memory>
#include <mutex>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/base/transfer.hh"
#include "jetstream/render/base/window_attachment.hh"

namespace Jetstream::Render {

class JETSTREAM_API Texture : public WindowAttachment {
 public:
    enum class PixelFormat : U64 {
        RGBA,
        RED,
    };

    enum class DataFormat : U64 {
        RGBA,
        UI8,
        F32,
    };

    enum class PixelType : U64 {
        UI8,
        F32,
    };

    struct Config {
        Extent2D<U64> size;
        const uint8_t* buffer = nullptr;
        DataFormat dfmt = DataFormat::RGBA;
        PixelFormat pfmt = PixelFormat::RGBA;
        PixelType ptype = PixelType::UI8;
        bool multisampled = false;
    };

    explicit Texture(const Config& config);
    virtual ~Texture() = default;

    Type type() const override {
        return Type::Texture;
    }

    const Config& getConfig() const {
        return config;
    }

    constexpr const bool& multisampled() const {
        return config.multisampled;
    }

    constexpr const Extent2D<U64>& size() const {
        return config.size;
    }
    bool size(const Extent2D<U64>& size);

    virtual uint64_t raw() const = 0;
    Result fill();
    Result fillRow(const U64& y, const U64& height);

    void presentationHint(U64 frame, bool visible) const {
        presentationVisible = (presentationFrame == frame && presentationVisible) || visible;
        presentationFrame = frame;
    }
    bool visibleForPresentation(U64 frame) const {
        return presentationFrame != frame || presentationVisible;
    }

    void markContentChanged() const {
        contentRevisionValue.fetch_add(1, std::memory_order_release);
    }
    U64 contentRevision() const {
        return contentRevisionValue.load(std::memory_order_acquire);
    }

    U64 allocation() const {
        return allocationValue.load(std::memory_order_acquire);
    }
    bool allocated() const {
        return allocationLive.load(std::memory_order_acquire);
    }

    template<DeviceType D>
    static std::shared_ptr<Texture> Factory(const Config& config) {
        return std::make_shared<TextureImp<D>>(config);
    }

 protected:
    Result validateFillRow(const U64& y, const U64& height) const;

    void markAllocated() {
        allocationLive.store(true, std::memory_order_release);
        allocationValue.fetch_add(1, std::memory_order_release);
    }
    void markDeallocated() {
        allocationLive.store(false, std::memory_order_release);
        allocationValue.fetch_add(1, std::memory_order_release);
    }

    Config config;

    U64 pixelByteSize() const;
 private:
    Result fillRowLocked(const U64& y, const U64& height);
    void restorePendingUploads(std::vector<Transfer::PendingUpload> uploads);

    std::mutex uploadMutex;
    Transfer::PendingUploadQueue pendingUploads;
    std::atomic<U64> uploadGeneration = 0;
    mutable U64 presentationFrame = ~U64{0};
    mutable bool presentationVisible = true;
    mutable std::atomic<U64> contentRevisionValue = 0;
    std::atomic<U64> allocationValue = 0;
    std::atomic<bool> allocationLive = false;

    friend class Transfer;
    friend class Transfer::Batch;
};

}  // namespace Jetstream::Render

#endif
