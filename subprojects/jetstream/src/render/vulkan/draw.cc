#include "jetstream/render/vulkan/vertex.hh"
#include "jetstream/render/vulkan/draw.hh"

namespace Jetstream::Render {

using Implementation = DrawImp<Device::Vulkan>;

Implementation::DrawImp(const Config& config) : Draw(config) {
    buffer = std::dynamic_pointer_cast<VertexImp<Device::Vulkan>>(config.buffer);
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating draw.");

    JST_CHECK(buffer->create());

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer,
                              const U64& offset) {
    // TODO: Draw type was here.

    JST_CHECK(buffer->encode(commandBuffer, offset));

    vkCmdDrawIndexed(commandBuffer, buffer->getIndicesCount(), 1, 0, 0, 0);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying draw.");

    JST_CHECK(buffer->destroy());

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
