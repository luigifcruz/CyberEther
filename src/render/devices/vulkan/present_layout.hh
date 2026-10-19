#ifndef JETSTREAM_RENDER_VULKAN_PRESENT_LAYOUT_HH
#define JETSTREAM_RENDER_VULKAN_PRESENT_LAYOUT_HH

#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "creation.hh"

#include <array>

namespace Jetstream::Render::Detail {

inline Result CreatePresentLayout(VkDevice device, VkDescriptorSetLayout& layout) {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = static_cast<U32>(bindings.size());
    info.pBindings = bindings.data();

    JST_VK_CHECK(CreateHandle(layout, [&](auto* output) {
        return vkCreateDescriptorSetLayout(device, &info, nullptr, output);
    }), [&]{
        JST_ERROR("[VULKAN] Can't create presentation descriptor set layout.");
    });

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Detail

#endif  // JETSTREAM_RENDER_VULKAN_PRESENT_LAYOUT_HH
