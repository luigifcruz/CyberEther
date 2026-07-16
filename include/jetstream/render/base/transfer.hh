#ifndef JETSTREAM_RENDER_BASE_TRANSFER_HH
#define JETSTREAM_RENDER_BASE_TRANSFER_HH

#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream::Render {

class Buffer;
class Texture;

class JETSTREAM_API Transfer {
 public:
    struct JETSTREAM_API PendingUpload {
        U64 start = 0;
        U64 unitByteSize = 0;
        U64 generation = 0;
        std::vector<U8> data;

        U64 extent() const;
    };

    class JETSTREAM_API PendingUploadQueue {
     public:
        Result queue(const U64& start,
                     const U64& count,
                     const U64& totalCount,
                     const U64& unitByteSize,
                     const U8* source,
                     const U64& generation = 0);

        std::vector<PendingUpload> take();
        Result restore(std::vector<PendingUpload> restored);
        void clear();
        bool empty() const;

     private:
        Result queueLocked(const U64& start,
                           const U64& unitByteSize,
                           const U8* source,
                           const U64& byteSize,
                           const U64& generation);

        mutable std::mutex mutex;
        std::vector<PendingUpload> uploads;
    };

    struct JETSTREAM_API BufferTransfer {
        std::shared_ptr<Buffer> destination;
        U64 destinationOffset = 0;
        PendingUpload upload;
    };

    struct JETSTREAM_API TextureTransfer {
        std::shared_ptr<Texture> destination;
        U64 destinationRow = 0;
        U64 rowByteSize = 0;
        Extent2D<U64> destinationSize;
        PendingUpload upload;
    };

    class JETSTREAM_API Batch {
     public:
        Batch();
        ~Batch();

        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;
        Batch(Batch&&) = delete;
        Batch& operator=(Batch&&) = delete;

        bool empty() const;
        bool contains(const std::shared_ptr<Buffer>& buffer) const;
        bool contains(const std::shared_ptr<Texture>& texture) const;
        void collect(const std::shared_ptr<Buffer>& buffer);
        void collect(const std::shared_ptr<Texture>& texture);
        void commit();

        const std::vector<BufferTransfer>& buffers() const;
        const std::vector<TextureTransfer>& textures() const;

     private:
        bool committed = false;
        std::vector<BufferTransfer> bufferTransfers;
        std::vector<TextureTransfer> textureTransfers;
        std::unordered_set<const Buffer*> collectedBuffers;
        std::unordered_set<const Texture*> collectedTextures;
        std::unordered_set<const Buffer*> pendingBuffers;
        std::unordered_set<const Texture*> pendingTextures;
    };

    Transfer();
    virtual ~Transfer();

 protected:
    static constexpr U64 MinimumBufferSize = 4 * 1024 * 1024;

    static bool reserveRange(U64& used,
                             const U64& size,
                             const U64& alignment,
                             U64& offset);
    static bool calculateCapacity(const U64& required,
                                  const U64& alignment,
                                  U64& capacity);
    static bool calculateAlignedSize(const U64& size,
                                     const U64& alignment,
                                     U64& alignedSize);
    static bool validateTexture(const TextureTransfer& transfer,
                                U64& rowCount);
    static bool copyBuffer(U8* destination,
                           const U8* source,
                           const U64& size);
    static bool copyTextureRows(U8* destination,
                                const U8* source,
                                const U64& rowCount,
                                const U64& rowByteSize,
                                const U64& encodedRowByteSize);
};

}  // namespace Jetstream::Render

#endif
