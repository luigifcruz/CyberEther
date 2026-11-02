#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <limits>
#include <utility>

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/draw.hh"
#include "jetstream/render/devices/metal/kernel.hh"
#include "jetstream/render/devices/metal/program.hh"
#include "jetstream/render/devices/metal/surface.hh"
#include "jetstream/render/devices/metal/texture.hh"
#include "jetstream/render/devices/metal/vertex.hh"
#endif

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
#include "jetstream/render/devices/vulkan/buffer.hh"
#include "jetstream/render/devices/vulkan/draw.hh"
#include "jetstream/render/devices/vulkan/kernel.hh"
#include "jetstream/render/devices/vulkan/program.hh"
#include "jetstream/render/devices/vulkan/surface.hh"
#include "jetstream/render/devices/vulkan/texture.hh"
#include "jetstream/render/devices/vulkan/vertex.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "../../../src/render/devices/vulkan/buffer_allocation.hh"
#include "../../../src/render/devices/vulkan/buffer_memory.hh"
#include "../../../src/render/devices/vulkan/creation.hh"
#endif

#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/draw.hh"
#include "jetstream/render/devices/webgpu/kernel.hh"
#include "jetstream/render/devices/webgpu/program.hh"
#include "jetstream/render/devices/webgpu/surface.hh"
#include "jetstream/render/devices/webgpu/texture.hh"
#include "jetstream/render/devices/webgpu/vertex.hh"
#endif

using namespace Jetstream;

namespace {

template<DeviceType D>
class ValidationSurface final : public Render::SurfaceImp<D> {
 public:
    using Render::SurfaceImp<D>::SurfaceImp;
    using Render::SurfaceImp<D>::validateAppend;
};

template<DeviceType D>
class HiddenSurface final : public Render::SurfaceImp<D> {
 public:
    using Render::SurfaceImp<D>::SurfaceImp;
    bool wantsDraw() { return this->shouldDraw(); }
    bool wantsGraphics(bool visible) { return this->shouldDrawGraphics(visible); }
    Result bindingResult() const { return this->drawResult(); }
    Result hiddenFrame() {
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
        if constexpr (D == DeviceType::Metal) {
            return this->draw(nullptr, false);
        }
#endif
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
        if constexpr (D == DeviceType::Vulkan) {
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            return this->encode(commandBuffer, false);
        }
#endif
#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
        if constexpr (D == DeviceType::WebGPU) {
            WGPUCommandEncoder commandEncoder = nullptr;
            return this->draw(commandEncoder, false);
        }
#endif
    }
};

template<DeviceType D>
class HiddenTexture final : public Render::TextureImp<D> {
 public:
    using Render::TextureImp<D>::TextureImp;
    void allocate() { this->markAllocated(); }
    void deallocate() { this->markDeallocated(); }
};

template<DeviceType D>
class HiddenProgram final : public Render::ProgramImp<D> {
 public:
    using Render::ProgramImp<D>::ProgramImp;
    Result rebindTextures() override {
        ++rebinds;
        return std::exchange(rebindResult, Result::SUCCESS);
    }
    U64 rebinds = 0;
    Result rebindResult = Result::SUCCESS;
};

template<DeviceType D>
void CheckHiddenCancelledGraphics() {
    auto framebuffer = std::make_shared<HiddenTexture<D>>(Render::Texture::Config{.size = {2, 2}});
    HiddenSurface<D> surface(Render::Surface::Config{.framebuffer = framebuffer, .retained = true});
    REQUIRE(surface.wantsDraw());
    REQUIRE(surface.wantsGraphics(true));
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());

    surface.invalidate();
    REQUIRE(surface.wantsDraw());
    REQUIRE(surface.wantsGraphics(true));
    const U64 revision = framebuffer->contentRevision();
    for (U64 frame = 0; frame < 3; ++frame) {
        REQUIRE(surface.hiddenFrame() == Result::SUCCESS);
        surface.commitDraw();
        REQUIRE(framebuffer->contentRevision() == revision);
        REQUIRE(surface.wantsDraw());
    }

    REQUIRE(surface.wantsGraphics(true));
    surface.commitDraw();
    REQUIRE(framebuffer->contentRevision() == revision + 1);
    REQUIRE_FALSE(surface.wantsDraw());
    REQUIRE(surface.hiddenFrame() == Result::SUCCESS);
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());
}

template<DeviceType D>
void CheckHiddenBorrowedTextures() {
    auto texture = std::make_shared<HiddenTexture<D>>(Render::Texture::Config{.size = {2, 2}});
    texture->allocate();
    auto program = std::make_shared<HiddenProgram<D>>(Render::Program::Config{
        .textures = {texture}, .borrowedTextures = true,
    });
    HiddenSurface<D> surface(Render::Surface::Config{.programs = {program}, .retained = true});
    const auto drawVisible = [&]() {
        REQUIRE(surface.wantsDraw());
        REQUIRE(surface.bindingResult() == Result::SUCCESS);
        REQUIRE(surface.wantsGraphics(true));
        surface.commitDraw();
        REQUIRE_FALSE(surface.wantsDraw());
    };
    drawVisible();
    REQUIRE(program->rebinds == 0);

    SECTION("A pending visible rebind failure is retried while hidden") {
        texture->deallocate();
        texture->allocate();
        program->rebindResult = Result::ERROR;
        REQUIRE_FALSE(surface.wantsDraw());
        REQUIRE(surface.bindingResult() == Result::ERROR);
        REQUIRE(program->rebinds == 1);
        REQUIRE(surface.hiddenFrame() == Result::SUCCESS);
        REQUIRE(program->rebinds == 2);
        surface.commitDraw();
        drawVisible();
    }
    SECTION("Hidden reallocation failures propagate and retry without losing redraw") {
        texture->deallocate();
        texture->allocate();
        program->rebindResult = Result::ERROR;
        REQUIRE(surface.hiddenFrame() == Result::ERROR);
        REQUIRE(surface.bindingResult() == Result::ERROR);
        REQUIRE(program->rebinds == 1);
        surface.commitDraw();
        REQUIRE(surface.hiddenFrame() == Result::SUCCESS);
        REQUIRE(program->rebinds == 2);
        surface.commitDraw();
        drawVisible();
    }
    SECTION("Hidden deallocation remains an error until the producer is live") {
        texture->deallocate();
        for (U64 frame = 0; frame < 2; ++frame) {
            REQUIRE(surface.hiddenFrame() == Result::ERROR);
            REQUIRE(surface.bindingResult() == Result::ERROR);
            REQUIRE(program->rebinds == 0);
            surface.commitDraw();
        }
        texture->allocate();
        REQUIRE(surface.hiddenFrame() == Result::SUCCESS);
        REQUIRE(program->rebinds == 1);
        surface.commitDraw();
        drawVisible();
    }
    SECTION("Hidden producer content changes preserve a later visible redraw") {
        texture->markContentChanged();
        REQUIRE(surface.hiddenFrame() == Result::SUCCESS);
        REQUIRE(program->rebinds == 0);
        surface.commitDraw();
        drawVisible();
    }
    REQUIRE(surface.hiddenFrame() == Result::SUCCESS);
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());
}

template<DeviceType D>
void CheckCompatibility() {
    auto texture = std::make_shared<Render::TextureImp<D>>(Render::Texture::Config{.size = {2, 2}});
    ValidationSurface<D> surface(Render::Surface::Config{.framebuffer = texture});
    auto buffer = std::make_shared<Render::BufferImp<D>>(Render::Buffer::Config{});
    auto vertex = std::make_shared<Render::VertexImp<D>>(Render::Vertex::Config{.vertices = {{buffer, 2}}});
    auto draw = std::make_shared<Render::DrawImp<D>>(Render::Draw::Config{.buffer = vertex});
    Render::Program::Config config;
    config.draws = {draw};
    config.textures = {texture};
    config.buffers = {{buffer, Render::Program::Target::VERTEX}};
    auto program = std::make_shared<Render::ProgramImp<D>>(config);
    auto kernel = std::make_shared<Render::KernelImp<D>>(Render::Kernel::Config{});
    REQUIRE(surface.validateAppend({program}, {kernel}) == Result::SUCCESS);
    REQUIRE(surface.validateAppend({program, nullptr}, {kernel}) == Result::ERROR);
    REQUIRE(surface.validateAppend({program}, {kernel, nullptr}) == Result::ERROR);
    REQUIRE(surface.validateAppend({std::make_shared<Render::Program>(config)}, {}) == Result::ERROR);
    REQUIRE(surface.validateAppend({}, {std::make_shared<Render::Kernel>(Render::Kernel::Config{})}) == Result::ERROR);

    config.draws = {nullptr};
    REQUIRE(surface.validateAppend({std::make_shared<Render::ProgramImp<D>>(config)}, {}) == Result::ERROR);
    config.draws = {std::make_shared<Render::DrawImp<D>>(Render::Draw::Config{})};
    REQUIRE(surface.validateAppend({std::make_shared<Render::ProgramImp<D>>(config)}, {}) == Result::ERROR);
    config.draws = {std::make_shared<Render::DrawImp<D>>(Render::Draw::Config{
        .buffer = std::make_shared<Render::Vertex>(Render::Vertex::Config{}),
    })};
    REQUIRE(surface.validateAppend({std::make_shared<Render::ProgramImp<D>>(config)}, {}) == Result::ERROR);
    config.draws = {draw};
    config.textures = {std::make_shared<Render::TextureImp<D>>(Render::Texture::Config{})};
    config.buffers = {{nullptr, Render::Program::Target::VERTEX}};
    REQUIRE(surface.validateAppend({std::make_shared<Render::ProgramImp<D>>(config)}, {}) == Result::ERROR);
    config.buffers = {{buffer, Render::Program::Target::VERTEX}};
    config.textures = {nullptr};
    REQUIRE(surface.validateAppend({std::make_shared<Render::ProgramImp<D>>(config)}, {}) == Result::ERROR);
    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers = {{nullptr, Render::Kernel::AccessMode::READ}};
    REQUIRE(surface.validateAppend({}, {std::make_shared<Render::KernelImp<D>>(kernelConfig)}) == Result::ERROR);
    REQUIRE(surface.validateAppend({program}, {kernel}) == Result::SUCCESS);
    REQUIRE(surface.getConfig().programs.empty());
    REQUIRE(surface.getConfig().kernels.empty());
}

}  // namespace

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
TEST_CASE("Metal hidden frames preserve cancelled visible redraws",
          "[core][render][surface][visibility][metal]") {
    CheckHiddenCancelledGraphics<DeviceType::Metal>();
}

TEST_CASE("Metal hidden frames refresh borrowed textures and propagate binding failures",
          "[core][render][surface][visibility][metal]") {
    CheckHiddenBorrowedTextures<DeviceType::Metal>();
}

TEST_CASE("Metal surface append validation rejects incompatible objects without backend allocation",
          "[core][render][surface][extend][metal]") {
    CheckCompatibility<DeviceType::Metal>();
}
#endif

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
TEST_CASE("Vulkan hidden frames preserve cancelled visible redraws",
          "[core][render][surface][visibility][vulkan]") {
    CheckHiddenCancelledGraphics<DeviceType::Vulkan>();
}

TEST_CASE("Vulkan hidden frames refresh borrowed textures and propagate binding failures",
          "[core][render][surface][visibility][vulkan]") {
    CheckHiddenBorrowedTextures<DeviceType::Vulkan>();
}

TEST_CASE("Vulkan surface append validation rejects incompatible objects without backend allocation",
          "[core][render][surface][extend][vulkan]") {
    CheckCompatibility<DeviceType::Vulkan>();
}

TEST_CASE("Vulkan buffer allocation failures only release initialized handles and allow retry",
          "[core][render][vulkan][surface][extend][faults]") {
    U64 failure = 0;
    bool throws = false;
    SECTION("Create failure") {}
    SECTION("Allocate failure") { failure = 1; }
    SECTION("Bind failure") { failure = 2; }
    SECTION("Create exception") { throws = true; }
    SECTION("Allocate exception") { failure = 1; throws = true; }
    SECTION("Bind exception") { failure = 2; throws = true; }
    const auto handle = []<typename T>(U64 value) {
        if constexpr (std::is_pointer_v<T>) {
            return reinterpret_cast<T>(static_cast<uintptr_t>(value));
        } else {
            return static_cast<T>(value);
        }
    };
    const VkBuffer validBuffer = handle.template operator()<VkBuffer>(1);
    const VkDeviceMemory validMemory = handle.template operator()<VkDeviceMemory>(2);
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    U64 destroyed = 0;
    U64 freed = 0;
    const auto cleanup = [&](VkBuffer created, VkDeviceMemory allocated) {
        REQUIRE((created == VK_NULL_HANDLE || created == validBuffer));
        REQUIRE((allocated == VK_NULL_HANDLE || allocated == validMemory));
        destroyed += created != VK_NULL_HANDLE;
        freed += allocated != VK_NULL_HANDLE;
    };
    const auto fail = [&](U64 stage) {
        if (stage != failure) {
            return Result::SUCCESS;
        }
        if (throws) {
            throw Result::ERROR;
        }
        return Result::ERROR;
    };
    const auto request = [&]() {
        return Render::Detail::AllocateBuffer(buffer, memory, [&](VkBuffer& candidate) {
            candidate = handle.template operator()<VkBuffer>(0xbebebebebebebebe);
            const Result result = fail(0);
            if (result == Result::SUCCESS) candidate = validBuffer;
            return result;
        }, [&](VkBuffer created, VkDeviceMemory& candidate) {
            REQUIRE(created == validBuffer);
            candidate = handle.template operator()<VkDeviceMemory>(0xbebebebebebebebe);
            const Result result = fail(1);
            if (result == Result::SUCCESS) candidate = validMemory;
            return result;
        }, [&](VkBuffer created, VkDeviceMemory allocated) {
            REQUIRE(created == validBuffer);
            REQUIRE(allocated == validMemory);
            return fail(2);
        }, cleanup);
    };
    if (throws) REQUIRE_THROWS_AS(request(), Result);
    else REQUIRE(request() == Result::ERROR);
    REQUIRE(buffer == VK_NULL_HANDLE);
    REQUIRE(memory == VK_NULL_HANDLE);
    REQUIRE(destroyed == (failure > 0 ? 1 : 0));
    REQUIRE(freed == (failure > 1 ? 1 : 0));
    failure = 3;
    REQUIRE(request() == Result::SUCCESS);
    REQUIRE(buffer == validBuffer);
    REQUIRE(memory == validMemory);
    cleanup(buffer, memory);
    Render::BufferImp<DeviceType::Vulkan> uncreated(Render::Buffer::Config{});
    REQUIRE(uncreated.destroy() == Result::SUCCESS);
    REQUIRE(uncreated.destroy() == Result::SUCCESS);
}

TEST_CASE("Vulkan buffer pages align disjoint ranges and coalesce retired allocations",
          "[core][render][vulkan][buffer-memory]") {
    Render::Detail::BufferMemoryRanges ranges(4096);
    const auto first = ranges.take(17, 1);
    const auto second = ranges.take(33, 256);
    const auto third = ranges.take(1000, 512);
    REQUIRE(first == 0);
    REQUIRE(second == 256);
    REQUIRE(third == 512);
    REQUIRE_FALSE(ranges.take(4096, 1));
    REQUIRE_FALSE(ranges.take(0, 1));
    REQUIRE_FALSE(ranges.take(1, 0));
    REQUIRE_FALSE(ranges.take(std::numeric_limits<VkDeviceSize>::max(), 256));
    ranges.release(*second, 33);
    REQUIRE(ranges.take(33, 256) == second);
    ranges.release(*first, 17);
    ranges.release(*third, 1000);
    ranges.release(*second, 33);
    REQUIRE(ranges.take(4096, 1) == 0);
    REQUIRE_FALSE(ranges.take(1, 1));
}

TEST_CASE("Vulkan buffer pages reuse interleaved frees without overlapping live buffers",
          "[core][render][vulkan][buffer-memory]") {
    Render::Detail::BufferMemoryRanges ranges(65536);
    std::vector<VkDeviceSize> offsets;
    for (size_t i = 0; i < 256; ++i) {
        const auto offset = ranges.take(256, 256);
        REQUIRE(offset == i * 256);
        offsets.push_back(*offset);
    }
    REQUIRE_FALSE(ranges.take(256, 256));
    for (size_t i = 0; i < offsets.size(); i += 2) {
        ranges.release(offsets[i], 256);
    }
    for (size_t i = 0; i < offsets.size(); i += 2) {
        REQUIRE(ranges.take(256, 256) == offsets[i]);
    }
    for (size_t i = offsets.size(); i-- > 0;) {
        ranges.release(offsets[i], 256);
    }
    REQUIRE(ranges.take(65536, 256) == 0);
}

TEST_CASE("Vulkan buffer memory shares pages and releases them after the last buffer",
          "[core][render][native][vulkan][buffer-memory-native]") {
    if (!std::getenv("JETSTREAM_TEST_NATIVE_COMPONENTS")) {
        SKIP("Set JETSTREAM_TEST_NATIVE_COMPONENTS=1 to run headless GPU regressions.");
    }
    const Backend::Config config{.validationEnabled = false, .headless = true};
    REQUIRE(Backend::Initialize<DeviceType::Vulkan>(config) == Result::SUCCESS);
    auto& backend = Backend::State<DeviceType::Vulkan>();
    const auto device = backend->getDevice();
    struct BoundBuffer {
        VkDevice device;
        VkBuffer buffer = VK_NULL_HANDLE;
        std::shared_ptr<Render::Detail::BufferMemory> memory;

        explicit BoundBuffer(VkDevice device) : device(device) {}

        ~BoundBuffer() {
            if (buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, buffer, nullptr);
            }
        }
    };
    const auto create = [&](bool dedicated) {
        auto buffer = std::make_unique<BoundBuffer>(device);
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = 1024;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        REQUIRE(vkCreateBuffer(device, &info, nullptr, &buffer->buffer) == VK_SUCCESS);
        VkMemoryDedicatedRequirements dedicatedRequirements{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        requirements.pNext = &dedicatedRequirements;
        VkBufferMemoryRequirementsInfo2 request{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};
        request.buffer = buffer->buffer;
        vkGetBufferMemoryRequirements2(device, &request, &requirements);
        const auto& required = requirements.memoryRequirements;
        const auto type = Backend::FindMemoryType(backend->getPhysicalDevice(), required.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        REQUIRE(Render::Detail::AllocateBufferMemory(device, buffer->buffer, type, required,
                    dedicated || dedicatedRequirements.requiresDedicatedAllocation, buffer->memory) == VK_SUCCESS);
        CHECK(buffer->memory->offset % required.alignment == 0);
        const auto& memory = *buffer->memory;
        REQUIRE(vkBindBufferMemory(device, buffer->buffer, memory.memory, memory.offset) == VK_SUCCESS);
        return buffer;
    };
    auto first = create(false);
    auto second = create(false);
    const auto block = std::weak_ptr(first->memory->block);
    if (first->memory->memory == second->memory->memory) {
        CHECK(second->memory->offset >= first->memory->offset + first->memory->size);
        const auto offset = first->memory->offset;
        first.reset();
        CHECK_FALSE(block.expired());
        auto reused = create(false);
        CHECK(reused->memory->memory == second->memory->memory);
        CHECK(reused->memory->offset == offset);
        auto dedicated = create(true);
        CHECK(dedicated->memory->memory != second->memory->memory);
        CHECK(dedicated->memory->offset == 0);
    }
    first.reset();
    second.reset();
    CHECK(block.expired());
}

TEST_CASE("Vulkan failed creation outputs never become owned rollback handles",
          "[core][render][vulkan][surface][extend][faults]") {
    const auto check = []<typename Handle>() {
        const auto handle = [](uintptr_t value) {
            if constexpr (std::is_pointer_v<Handle>) return reinterpret_cast<Handle>(value);
            else return static_cast<Handle>(value);
        };
        const Handle poison = handle(0xbebebebebebebebe);
        const Handle valid = handle(1);
        Handle owned = VK_NULL_HANDLE;
        for (const VkResult failure : {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY}) {
            REQUIRE(Render::Detail::CreateHandle(owned, [&](Handle* output) {
                *output = poison;
                return failure;
            }) == failure);
            REQUIRE(owned == VK_NULL_HANDLE);
            REQUIRE_THROWS_AS(Render::Detail::CreateHandle(owned, [&](Handle* output) -> VkResult {
                *output = poison;
                throw Result::ERROR;
            }), Result);
            REQUIRE(owned == VK_NULL_HANDLE);
        }
        REQUIRE(Render::Detail::CreateHandle(owned, [&](Handle* output) {
            *output = valid;
            return VK_SUCCESS;
        }) == VK_SUCCESS);
        REQUIRE(owned == valid);
        REQUIRE(Render::Detail::CreateHandle(owned, [&](Handle* output) {
            *output = poison;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }) == VK_ERROR_OUT_OF_HOST_MEMORY);
        REQUIRE(owned == valid);
    };
    check.template operator()<VkPipelineLayout>();
    check.template operator()<VkDescriptorSetLayout>();
    check.template operator()<VkDescriptorSet>();
    check.template operator()<VkPipeline>();
    check.template operator()<VkShaderModule>();
    check.template operator()<VkFramebuffer>();
    check.template operator()<VkRenderPass>();
    check.template operator()<VkImage>();
    check.template operator()<VkImageView>();
    check.template operator()<VkSampler>();
    check.template operator()<VkDeviceMemory>();
}
#endif

#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
TEST_CASE("WebGPU hidden frames preserve cancelled visible redraws",
          "[core][render][surface][visibility][webgpu]") {
    CheckHiddenCancelledGraphics<DeviceType::WebGPU>();
}

TEST_CASE("WebGPU hidden frames refresh borrowed textures and propagate binding failures",
          "[core][render][surface][visibility][webgpu]") {
    CheckHiddenBorrowedTextures<DeviceType::WebGPU>();
}

TEST_CASE("WebGPU surface append validation rejects incompatible objects without backend allocation",
          "[core][render][surface][extend][webgpu]") {
    CheckCompatibility<DeviceType::WebGPU>();
}

TEST_CASE("WebGPU surface extensions fall back before allocating unvalidated candidates",
          "[core][render][webgpu][surface][extend]") {
    class Buffer final : public Render::BufferImp<DeviceType::WebGPU> {
     public:
        Buffer() : BufferImp(Render::Buffer::Config{}) {}
        Result create() override { ++creates; return Result::SUCCESS; }
        Result destroy() override { ++destroys; return Result::SUCCESS; }
        U64 creates = 0;
        U64 destroys = 0;
    };
    class Surface final : public Render::SurfaceImp<DeviceType::WebGPU> {
     public:
        explicit Surface(const Config& config) : SurfaceImp(config) {}
        Result create() override { resetFramebuffer(); return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        bool wantsDraw() { return shouldDraw(); }
        using SurfaceImp::createAppended;
    };
    class Window final : public Render::Window {
     public:
        Window() : Render::Window(Config{}) {}
        std::string info() const override { return "fallback"; }
        constexpr DeviceType device() const override { return DeviceType::WebGPU; }

     protected:
        Result bindSurface(const std::shared_ptr<Render::Surface>&) override { return Result::SUCCESS; }
        Result unbindSurface(const std::shared_ptr<Render::Surface>&) override { return Result::SUCCESS; }
        Result underlyingCreate() override { return Result::SUCCESS; }
        Result underlyingDestroy() override { return Result::SUCCESS; }
        Result underlyingBegin() override { return Result::SUCCESS; }
        Result underlyingEnd() override { return Result::SUCCESS; }
        Result underlyingSynchronize() override { return Result::SUCCESS; }

    };
    Window window;
    auto surface = std::make_shared<Surface>(Render::Surface::Config{.retained = true});
    auto buffer = std::make_shared<Buffer>();
    Render::Program::Config config;
    config.buffers = {{buffer, Render::Program::Target::VERTEX}};
    config.shaders[DeviceType::WebGPU] = {{'i', 'n', 'v', 'a', 'l', 'i', 'd'}, {1}};
    auto program = std::make_shared<Render::ProgramImp<DeviceType::WebGPU>>(config);
    auto kernel = std::make_shared<Render::KernelImp<DeviceType::WebGPU>>(Render::Kernel::Config{});
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();
    REQUIRE_FALSE(surface->wantsDraw());
    REQUIRE(surface->createAppended({program}, {kernel}) == Result::SKIP);
    for (U64 retry = 0; retry < 3; ++retry) {
        REQUIRE(window.extend(surface, {program}, {kernel}) == Result::SKIP);
        REQUIRE(surface->getConfig().programs.empty());
        REQUIRE(surface->getConfig().kernels.empty());
        REQUIRE(buffer->creates == 0);
        REQUIRE(buffer->destroys == 0);
        REQUIRE_FALSE(surface->wantsDraw());
    }
    REQUIRE(window.unbind(surface) == Result::SUCCESS);
    auto replacement = std::make_shared<Surface>(Render::Surface::Config{
        .kernels = {kernel}, .programs = {program}, .retained = true,
    });
    REQUIRE(window.bind(replacement) == Result::SUCCESS);
    REQUIRE(buffer->creates == 1);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(buffer->destroys == 1);
}
#endif
