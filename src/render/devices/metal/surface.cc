#include "jetstream/render/devices/metal/program.hh"
#include "jetstream/render/devices/metal/texture.hh"
#include "jetstream/render/devices/metal/surface.hh"
#include "jetstream/render/devices/metal/kernel.hh"
#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/draw.hh"
#include "jetstream/render/devices/metal/vertex.hh"

namespace Jetstream::Render {

using Implementation = SurfaceImp<DeviceType::Metal>;

Implementation::SurfaceImp(const Config& config) : Surface(config) {
    framebufferResolve = std::dynamic_pointer_cast<
        TextureImp<DeviceType::Metal>>(config.framebuffer);

    if (config.multisampled) {
        auto framebufferConfig = framebufferResolve->getConfig();
        framebufferConfig.multisampled = true;
        framebufferConfig.buffer = nullptr;
        framebuffer = std::make_shared<TextureImp<DeviceType::Metal>>(framebufferConfig);
    }

    (void)reorderPrograms(config.programs);

    for (auto& kernel : config.kernels) {
        kernels.push_back(
            std::dynamic_pointer_cast<KernelImp<DeviceType::Metal>>(kernel)
        );
    }
}

Result Implementation::reorderPrograms(const std::vector<std::shared_ptr<Program>>& ordered) {
    decltype(programs) candidate;
    for (const auto& program : ordered) {
        auto typed = std::dynamic_pointer_cast<ProgramImp<DeviceType::Metal>>(program);
        if (!typed) {
            return Result::ERROR;
        }
        candidate.push_back(std::move(typed));
    }
    programs.swap(candidate);
    return Result::SUCCESS;
}

bool Implementation::appendable() const {
    return true;
}

Result Implementation::validateAppend(const std::vector<std::shared_ptr<Program>>& newPrograms,
                                       const std::vector<std::shared_ptr<Kernel>>& newKernels) const {
    return validateObjects<DeviceType::Metal>(newPrograms, newKernels);
}

Result Implementation::createAppended(const std::vector<std::shared_ptr<Program>>& newPrograms,
                                       const std::vector<std::shared_ptr<Kernel>>& newKernels) {
    const auto& activeFramebuffer = (config.multisampled) ? framebuffer : framebufferResolve;
    return appendObjects(newPrograms, newKernels, programs, kernels, [&](auto& object) {
        if constexpr (std::is_same_v<std::decay_t<decltype(object)>,
                                     std::shared_ptr<ProgramImp<DeviceType::Metal>>>) {
            return object->create(activeFramebuffer->getPixelFormat(), config.multisampled);
        } else {
            return object->create();
        }
    }, [](auto& object) { return object->destroy(); });
}

Result Implementation::create() {
    JST_DEBUG("[METAL] Creating surface.");

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor, "Failed to create render pass descriptor.");

    JST_CHECK(createFramebuffer());

    const auto& activeFramebuffer = (config.multisampled) ? framebuffer : framebufferResolve;
    for (auto& program : programs) {
        JST_CHECK(program->create(activeFramebuffer->getPixelFormat(), config.multisampled));
    }

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->create());
    }

    requestedSize = framebufferResolve->size();
    resetFramebuffer();

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[METAL] Destroying surface.");

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->destroy());
    }

    for (auto& program : programs) {
        JST_CHECK(program->destroy());
    }

    JST_CHECK(destroyFramebuffer());

    if (renderPassDescriptor) {
        renderPassDescriptor->release();
        renderPassDescriptor = nullptr;
    }

    return Result::SUCCESS;
}

Result Implementation::createFramebuffer() {
    JST_DEBUG("[METAL] Creating surface framebuffer.");

    if (config.multisampled) {
        JST_CHECK(framebuffer->create());
    }

    const auto clearColor = MTL::ClearColor(config.clearColor.r,
                                            config.clearColor.g,
                                            config.clearColor.b,
                                            config.clearColor.a);

    auto colorAttachDescOff = renderPassDescriptor->colorAttachments()->object(0)->init();

    if (config.multisampled) {
        auto textureResolve = reinterpret_cast<const MTL::Texture*>(framebufferResolve->raw());
        auto texture = reinterpret_cast<const MTL::Texture*>(framebuffer->raw());
        colorAttachDescOff->setTexture(texture);
        colorAttachDescOff->setResolveTexture(textureResolve);
        colorAttachDescOff->setLoadAction(MTL::LoadActionClear);
        colorAttachDescOff->setStoreAction(MTL::StoreActionMultisampleResolve);
        colorAttachDescOff->setClearColor(clearColor);
    } else {
        auto texture = reinterpret_cast<const MTL::Texture*>(framebufferResolve->raw());
        colorAttachDescOff->setTexture(texture);
        colorAttachDescOff->setLoadAction(MTL::LoadActionClear);
        colorAttachDescOff->setStoreAction(MTL::StoreActionStore);
        colorAttachDescOff->setClearColor(clearColor);
    }

    return Result::SUCCESS;
}

Result Implementation::destroyFramebuffer() {
    JST_DEBUG("[METAL] Destroying surface framebuffer");

    if (config.multisampled) {
        JST_CHECK(framebuffer->destroy());
    }

    return Result::SUCCESS;
}

Result Implementation::prepare() {
    framebufferChanged = framebufferResolve->size(requestedSize);
    if (framebufferChanged) {
        resetFramebuffer();
        if (config.multisampled) {
            framebuffer->size(requestedSize);
        }

        JST_CHECK(destroyFramebuffer());
        JST_CHECK(framebufferResolve->destroy());
        JST_CHECK(framebufferResolve->create());
        JST_CHECK(createFramebuffer());
    }

    return Result::SUCCESS;
}

Result Implementation::draw(MTL::CommandBuffer* commandBuffer, bool visible) {
    const bool required = shouldDraw(framebufferChanged);
    JST_CHECK(drawResult());
    if (!required) {
        return Result::SUCCESS;
    }
    if (hiddenAndIdle(visible, framebufferChanged)) {
        shouldDrawGraphics(visible);
        return Result::SUCCESS;
    }

    // Encode kernels.

    auto computeCmdEncoder = commandBuffer->computeCommandEncoder();
    for (auto& kernel : kernels) {
        JST_CHECK(kernel->encode(computeCmdEncoder));
    }
    computeCmdEncoder->endEncoding();

    if (!shouldDrawGraphics(visible)) {
        return Result::SUCCESS;
    }

    // Update clear color.

    const auto clearColor = MTL::ClearColor(config.clearColor.r,
                                            config.clearColor.g,
                                            config.clearColor.b,
                                            config.clearColor.a);
    renderPassDescriptor->colorAttachments()->object(0)->setClearColor(clearColor);

    // Encode programs.

    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor);
    const auto& sz = framebufferResolve->size();
    MTL::ScissorRect fullScissor = {0, 0, static_cast<NS::UInteger>(sz.x),
                                          static_cast<NS::UInteger>(sz.y)};
    for (auto& program : programs) {
        if (!program->enabled()) {
            continue;
        }
        renderCmdEncoder->setScissorRect(fullScissor);
        JST_CHECK(program->draw(renderCmdEncoder));
    }
    renderCmdEncoder->endEncoding();

    return Result::SUCCESS;
}

const Extent2D<U64>& Implementation::size(const Extent2D<U64>& size) {
    if (!framebufferResolve) {
        return NullSize2D;
    }

    requestedSize = size;
    invalidate();

    return framebufferResolve->size();
}

}  // namespace Jetstream::Render
