#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <new>
#include <thread>

#include "jetstream/render/devices/metal/program.hh"
#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/draw.hh"
#include "jetstream/render/devices/metal/vertex.hh"
#include "../../../src/render/devices/metal/program_cache.hh"

#include <objc/runtime.h>

using namespace Jetstream;
using namespace Jetstream::Render;
using Jetstream::Render::Detail::MetalProgramCache;

namespace {

thread_local int allocationCountdown = -1;

struct AllocationFailure {
    explicit AllocationFailure(int countdown) { allocationCountdown = countdown; }
    ~AllocationFailure() { allocationCountdown = -1; }
};

const std::string prefix = "#include <metal_stdlib>\nusing namespace metal;\n";
const std::string vertexSource = prefix +
    "vertex float4 main0(uint id [[vertex_id]]) { float2 p[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)}; return float4(p[id],0,1); }";
const std::string fragmentSource = prefix + "fragment float4 main0() { return float4(1,0,0,1); }";

std::vector<U8> Bytes(const std::string& text) { return {text.begin(), text.end()}; }

NS::SharedPtr<MTL::Function> Compile(MTL::Device* device, const std::string& source) {
    NS::Error* error = nullptr;
    auto library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nullptr, &error));
    REQUIRE(library);
    return NS::TransferPtr(library->newFunction(NS::String::string("main0", NS::UTF8StringEncoding)));
}

struct Objects {
    NS::SharedPtr<NS::AutoreleasePool> pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    NS::SharedPtr<MTL::Device> device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    NS::SharedPtr<MTL::Function> vertex;
    NS::SharedPtr<MTL::Function> fragment;
    NS::SharedPtr<MTL::RenderPipelineState> pipeline;

    Objects() {
        REQUIRE(device);
        vertex = Compile(device.get(), vertexSource);
        fragment = Compile(device.get(), fragmentSource);
        auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
        descriptor->setVertexFunction(vertex.get());
        descriptor->setFragmentFunction(fragment.get());
        descriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
        NS::Error* error = nullptr;
        pipeline = NS::TransferPtr(device->newRenderPipelineState(descriptor.get(), &error));
        REQUIRE(pipeline);
    }
};

struct TestProgram : ProgramImp<DeviceType::Metal> {
    using ProgramImp::ProgramImp;
    using ProgramImp::create;
    using ProgramImp::destroy;
    using ProgramImp::draw;
};

struct TestDraw : DrawImp<DeviceType::Metal> {
    using DrawImp::DrawImp;
    U64 resources() const { return transferBuffers.size(); }
};

Program::Config Config(U32 stride = 0, U32 instanceStride = 0, U32 uniformCount = 0) {
    Program::Config config;
    config.shaders[DeviceType::Metal] = {Bytes(vertexSource), Bytes(fragmentSource)};
    if (stride) {
        auto buffer = std::make_shared<BufferImp<DeviceType::Metal>>(Buffer::Config{
            .size = 48, .target = Buffer::Target::VERTEX, .elementByteSize = sizeof(F32),
        });
        Vertex::Config vertex{.vertices = {{buffer, stride}}};
        if (instanceStride) vertex.instances = {{buffer, instanceStride}};
        auto layout = std::make_shared<VertexImp<DeviceType::Metal>>(vertex);
        config.draws = {std::make_shared<TestDraw>(Draw::Config{.buffer = layout})};
        for (U32 i = 0; i < uniformCount; ++i) config.buffers.push_back({buffer, Program::Target::VERTEX});
    }
    return config;
}

struct Hooks;
Hooks* hooks = nullptr;
id NewLibrary(id, SEL, id, id, id*);
id NewFunction(id, SEL, id);
id NewPipeline(id, SEL, id, id*);

struct Hooks {
    Method libraryMethod;
    Method pipelineMethod;
    Method functionMethod = nullptr;
    IMP libraryOriginal;
    IMP pipelineOriginal;
    IMP functionOriginal = nullptr;
    U64 libraries = 0;
    U64 pipelines = 0;
    bool failFunction = false;
    bool failPipeline = false;
    NS::SharedPtr<MTL::Library> observedLibrary;
    NS::SharedPtr<MTL::Function> observedFunction;
    NS::SharedPtr<MTL::RenderPipelineState> observedPipeline;
    NS::UInteger beforeFailure = 0;

    explicit Hooks(MTL::Device* device) {
        hooks = this;
        const auto cls = object_getClass(reinterpret_cast<id>(device));
        libraryMethod = class_getInstanceMethod(cls, sel_registerName("newLibraryWithSource:options:error:"));
        pipelineMethod = class_getInstanceMethod(cls, sel_registerName("newRenderPipelineStateWithDescriptor:error:"));
        REQUIRE(libraryMethod);
        REQUIRE(pipelineMethod);
        libraryOriginal = method_setImplementation(libraryMethod, reinterpret_cast<IMP>(NewLibrary));
        pipelineOriginal = method_setImplementation(pipelineMethod, reinterpret_cast<IMP>(NewPipeline));
    }

    ~Hooks() {
        allocationCountdown = -1;
        method_setImplementation(libraryMethod, libraryOriginal);
        method_setImplementation(pipelineMethod, pipelineOriginal);
        if (functionMethod) method_setImplementation(functionMethod, functionOriginal);
        hooks = nullptr;
    }
};

id NewLibrary(id object, SEL selector, id source, id options, id* error) {
    ++hooks->libraries;
    auto library = reinterpret_cast<id(*)(id, SEL, id, id, id*)>(hooks->libraryOriginal)(object, selector, source, options, error);
    if (library && hooks->failFunction) {
        hooks->observedLibrary = NS::RetainPtr(reinterpret_cast<MTL::Library*>(library));
    }
    if (library && !hooks->functionMethod) {
        hooks->functionMethod = class_getInstanceMethod(object_getClass(library), sel_registerName("newFunctionWithName:"));
        hooks->functionOriginal = method_setImplementation(hooks->functionMethod, reinterpret_cast<IMP>(NewFunction));
    }
    return library;
}

id NewFunction(id object, SEL selector, id name) {
    auto function = reinterpret_cast<id(*)(id, SEL, id)>(hooks->functionOriginal)(object, selector, name);
    if (function && hooks->failFunction) {
        hooks->failFunction = false;
        hooks->observedFunction = NS::RetainPtr(reinterpret_cast<MTL::Function*>(function));
        hooks->beforeFailure = hooks->observedFunction->retainCount();
        allocationCountdown = 0;
    }
    return function;
}

id NewPipeline(id object, SEL selector, id descriptor, id* error) {
    ++hooks->pipelines;
    auto pipeline = reinterpret_cast<id(*)(id, SEL, id, id*)>(hooks->pipelineOriginal)(object, selector, descriptor, error);
    if (pipeline && hooks->failPipeline) {
        hooks->failPipeline = false;
        hooks->observedPipeline = NS::RetainPtr(reinterpret_cast<MTL::RenderPipelineState*>(pipeline));
        hooks->beforeFailure = hooks->observedPipeline->retainCount();
        allocationCountdown = 0;
    }
    return pipeline;
}

void RenderActive(MTL::Device* device, TestProgram& program) {
    auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, 4, 4, false)->copy());
    descriptor->setUsage(MTL::TextureUsageRenderTarget);
    auto texture = NS::TransferPtr(device->newTexture(descriptor.get()));
    auto queue = NS::TransferPtr(device->newCommandQueue());
    auto readback = NS::TransferPtr(device->newBuffer(1024, MTL::ResourceStorageModeShared));
    auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
    pass->colorAttachments()->object(0)->setTexture(texture.get());
    pass->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    pass->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
    auto command = queue->commandBuffer();
    auto encoder = command->renderCommandEncoder(pass);
    REQUIRE(program.draw(encoder) == Result::SUCCESS);
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    encoder->endEncoding();
    auto blit = command->blitCommandEncoder();
    blit->copyFromTexture(texture.get(), 0, 0, MTL::Origin(0, 0, 0), MTL::Size(4, 4, 1), readback.get(), 0, 256, 1024);
    blit->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    REQUIRE(command->status() == MTL::CommandBufferStatusCompleted);
    const auto pixels = static_cast<const U8*>(readback->contents());
    CHECK(pixels[0] == 255);
    CHECK(pixels[1] == 0);
    CHECK(pixels[2] == 0);
    CHECK(pixels[3] == 255);
}

}  // namespace

void* operator new(std::size_t size) {
    if (allocationCountdown >= 0 && allocationCountdown-- == 0) throw std::bad_alloc();
    if (auto memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }

TEST_CASE("Metal cache retains and bounds simulated device identities", "[metal][cache]") {
    Objects objects;
    MetalProgramCache cache;
    std::vector<NS::SharedPtr<NS::String>> identities;
    for (U64 i = 0; i < MetalProgramCache::MaxDevices + 2; ++i) {
        const auto name = std::string(64, 'x') + std::to_string(i);
        identities.push_back(NS::TransferPtr(NS::String::alloc()->init(name.c_str(), NS::UTF8StringEncoding)));
    }
    const auto identity = [&](U64 i) { return reinterpret_cast<MTL::Device*>(identities[i].get()); };
    const auto retained = identities[0]->retainCount();
    auto& first = cache.forDevice(identity(0));
    first.insertFunction("source", objects.vertex);
    first.insertPipeline("pipeline", objects.vertex, objects.fragment, objects.pipeline);
    CHECK(identities[0]->retainCount() == retained + 1);
    auto& second = cache.forDevice(identity(1));
    CHECK_FALSE(second.findFunction("source"));
    CHECK_FALSE(second.findPipeline("pipeline"));
    second.insertFunction("source", objects.fragment);
    CHECK(cache.forDevice(identity(0)).findFunction("source").get() == objects.vertex.get());
    CHECK(cache.forDevice(identity(1)).findFunction("source").get() == objects.fragment.get());
    for (U64 i = 2; i < identities.size(); ++i) {
        cache.forDevice(identity(i));
        CHECK(cache.devices.size() <= MetalProgramCache::MaxDevices);
    }
    CHECK(identities[0]->retainCount() == retained);
    CHECK_FALSE(cache.forDevice(identity(0)).findFunction("source"));
    CHECK_FALSE(cache.forDevice(identity(0)).findPipeline("pipeline"));
}

TEST_CASE("Metal cache eviction and teardown release owned functions and pipelines", "[metal][cache]") {
    Objects objects;
    const auto vertexCount = objects.vertex->retainCount();
    const auto fragmentCount = objects.fragment->retainCount();
    const auto pipelineCount = objects.pipeline->retainCount();
    const auto deviceCount = objects.device->retainCount();
    {
        MetalProgramCache cache;
        auto& entries = cache.forDevice(objects.device.get());
        entries.insertFunction("vertex", objects.vertex);
        entries.insertFunction("fragment", objects.fragment);
        entries.insertPipeline("active", objects.vertex, objects.fragment, objects.pipeline);
        auto active = entries.findPipeline("active");
        for (U64 i = 0; i < MetalProgramCache::MaxFunctions + 1; ++i) {
            entries.insertFunction(std::to_string(i), objects.fragment);
        }
        CHECK(entries.functions.size() == MetalProgramCache::MaxFunctions);
        CHECK_FALSE(entries.findFunction("vertex"));
        CHECK(objects.vertex->retainCount() == vertexCount + 1);
        CHECK(entries.findPipeline("active").get() == objects.pipeline.get());
        CHECK(objects.pipeline->retainCount() == pipelineCount + 2);
        for (U64 i = 0; i < MetalProgramCache::MaxPipelines + 1; ++i) {
            entries.insertPipeline(std::to_string(i), objects.fragment, objects.fragment, objects.pipeline);
        }
        CHECK(entries.pipelines.size() == MetalProgramCache::MaxPipelines);
        CHECK_FALSE(entries.findPipeline("active"));
        CHECK(objects.vertex->retainCount() == vertexCount);
        CHECK(active.get() == objects.pipeline.get());
        CHECK(active->device() == objects.device.get());
    }
    CHECK(objects.vertex->retainCount() == vertexCount);
    CHECK(objects.fragment->retainCount() == fragmentCount);
    CHECK(objects.pipeline->retainCount() == pipelineCount);
    CHECK(objects.device->retainCount() == deviceCount);
}

TEST_CASE("Metal cache failed node and bucket allocations preserve ownership and allow retry", "[metal][cache][allocation]") {
    Objects objects;
    for (int allocation = 0; allocation < 2; ++allocation) {
        MetalProgramCache cache;
        auto& entries = cache.forDevice(objects.device.get());
        const auto vertexCount = objects.vertex->retainCount();
        const auto fragmentCount = objects.fragment->retainCount();
        const auto pipelineCount = objects.pipeline->retainCount();
        bool threw = false;
        {
            AllocationFailure failure(allocation);
            try { entries.insertFunction("vertex", NS::RetainPtr(objects.vertex.get())); }
            catch (const std::bad_alloc&) { threw = true; }
        }
        REQUIRE(threw);
        CHECK(entries.functions.empty());
        CHECK(objects.vertex->retainCount() == vertexCount);
        entries.insertFunction("vertex", objects.vertex);
        REQUIRE(entries.findFunction("vertex").get() == objects.vertex.get());
        entries.functions.clear();
        threw = false;
        {
            AllocationFailure failure(allocation);
            try { entries.insertPipeline("pipeline", objects.vertex, objects.fragment, NS::RetainPtr(objects.pipeline.get())); }
            catch (const std::bad_alloc&) { threw = true; }
        }
        REQUIRE(threw);
        CHECK(entries.pipelines.empty());
        CHECK(objects.vertex->retainCount() == vertexCount);
        CHECK(objects.fragment->retainCount() == fragmentCount);
        CHECK(objects.pipeline->retainCount() == pipelineCount);
        entries.insertPipeline("pipeline", objects.vertex, objects.fragment, objects.pipeline);
        REQUIRE(entries.findPipeline("pipeline").get() == objects.pipeline.get());
    }
    MetalProgramCache cache;
    const auto deviceCount = objects.device->retainCount();
    bool threw = false;
    {
        AllocationFailure failure(0);
        try { cache.forDevice(objects.device.get()); }
        catch (const std::bad_alloc&) { threw = true; }
    }
    REQUIRE(threw);
    CHECK(cache.devices.empty());
    CHECK(objects.device->retainCount() == deviceCount);
    auto& entries = cache.forDevice(objects.device.get());
    for (U64 i = 0; i < MetalProgramCache::MaxPipelines; ++i) {
        const auto key = std::to_string(i);
        if (i < MetalProgramCache::MaxFunctions) entries.insertFunction(key, objects.vertex);
        entries.insertPipeline(key, objects.vertex, objects.fragment, objects.pipeline);
    }
    const auto vertexCount = objects.vertex->retainCount();
    const auto pipelineCount = objects.pipeline->retainCount();
    for (bool pipeline : {false, true}) {
        threw = false;
        {
            AllocationFailure failure(0);
            try {
                if (pipeline) entries.insertPipeline("new", objects.vertex, objects.fragment, objects.pipeline);
                else entries.insertFunction("new", objects.vertex);
            } catch (const std::bad_alloc&) { threw = true; }
        }
        REQUIRE(threw);
        CHECK(entries.functions.size() == MetalProgramCache::MaxFunctions);
        CHECK(entries.pipelines.size() == MetalProgramCache::MaxPipelines);
        CHECK(entries.functions.contains("0"));
        CHECK(entries.pipelines.contains("0"));
        CHECK(objects.vertex->retainCount() == vertexCount);
        CHECK(objects.pipeline->retainCount() == pipelineCount);
    }
}

TEST_CASE("Metal programs reuse complete keys, retry allocation failures and survive eviction", "[metal][cache][backend]") {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    REQUIRE(Backend::Initialize<DeviceType::Metal>(Backend::Config{.headless = true}) == Result::SUCCESS);
    struct Shutdown {
        ~Shutdown() { Backend::Destroy<DeviceType::Metal>(); }
    } shutdown;
    auto device = Backend::State<DeviceType::Metal>()->getDevice();
    Hooks counters(device);
    auto config = Config(2);
    {
        TestProgram first(config);
        REQUIRE(first.create(MTL::PixelFormatRGBA8Unorm, false) == Result::SUCCESS);
        REQUIRE(first.destroy() == Result::SUCCESS);
    }
    CHECK(counters.libraries == 2);
    CHECK(counters.pipelines == 1);
    {
        auto fresh = Config(2);
        TestProgram hit(fresh);
        REQUIRE(hit.create(MTL::PixelFormatRGBA8Unorm, false) == Result::SUCCESS);
        CHECK(std::static_pointer_cast<TestDraw>(fresh.draws[0])->resources() == 1);
        CHECK(fresh.draws[0]->updateVertexCount(3) == Result::SUCCESS);
        REQUIRE(hit.destroy() == Result::SUCCESS);
    }
    CHECK(counters.libraries == 2);
    CHECK(counters.pipelines == 1);
    std::array<std::thread, 8> workers;
    std::array<bool, 8> results{};
    for (U64 i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i] {
            auto threadPool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
            TestProgram program(Config(2));
            results[i] = program.create(MTL::PixelFormatRGBA8Unorm, false) == Result::SUCCESS;
            results[i] = program.destroy() == Result::SUCCESS && results[i];
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto result : results) CHECK(result);
    CHECK(counters.libraries == 2);
    CHECK(counters.pipelines == 1);
    for (int variation = 0; variation < 9; ++variation) {
        auto next = Config(variation == 0 ? 3 : 2, variation == 1 ? 4 : 0, variation == 2 ? 1 : 0);
        next.enableAlphaBlending = variation == 3;
        next.eraseBlending = variation == 4;
        if (variation == 7) next.shaders[DeviceType::Metal][0].push_back('\n');
        if (variation == 8) next.shaders[DeviceType::Metal][1].push_back('\n');
        const auto pipelines = counters.pipelines;
        TestProgram miss(next);
        REQUIRE(miss.create(variation == 5 ? MTL::PixelFormatRGBA32Float : MTL::PixelFormatRGBA8Unorm, variation == 6) == Result::SUCCESS);
        CHECK(counters.pipelines == pipelines + 1);
        REQUIRE(miss.destroy() == Result::SUCCESS);
    }
    CHECK(counters.libraries == 4);
    auto invalid = Config();
    invalid.shaders[DeviceType::Metal][1] = Bytes("invalid Metal shader");
    for (int retry = 0; retry < 2; ++retry) {
        TestProgram program(invalid);
        REQUIRE(program.create(MTL::PixelFormatRGBA8Unorm, false) == Result::ERROR);
        REQUIRE(program.destroy() == Result::SUCCESS);
    }
    auto uncached = Config();
    uncached.shaders[DeviceType::Metal][1].push_back(' ');
    TestProgram functionFailure(uncached);
    counters.failFunction = true;
    bool threw = false;
    try { functionFailure.create(MTL::PixelFormatRGBA8Unorm, false); }
    catch (const std::bad_alloc&) { threw = true; }
    allocationCountdown = -1;
    REQUIRE(threw);
    REQUIRE(counters.observedFunction);
    CHECK(counters.observedFunction->retainCount() + 1 == counters.beforeFailure);
    REQUIRE(functionFailure.destroy() == Result::SUCCESS);
    const auto libraries = counters.libraries;
    REQUIRE(functionFailure.create(MTL::PixelFormatRGBA8Unorm, false) == Result::SUCCESS);
    CHECK(counters.libraries == libraries + 1);
    REQUIRE(functionFailure.destroy() == Result::SUCCESS);
    TestProgram pipelineFailure(Config());
    counters.failPipeline = true;
    threw = false;
    try { pipelineFailure.create(MTL::PixelFormatRGBA32Float, false); }
    catch (const std::bad_alloc&) { threw = true; }
    allocationCountdown = -1;
    REQUIRE(threw);
    REQUIRE(counters.observedPipeline);
    CHECK(counters.observedPipeline->retainCount() + 1 == counters.beforeFailure);
    REQUIRE(pipelineFailure.destroy() == Result::SUCCESS);
    const auto pipelines = counters.pipelines;
    REQUIRE(pipelineFailure.create(MTL::PixelFormatRGBA32Float, false) == Result::SUCCESS);
    CHECK(counters.pipelines == pipelines + 1);
    REQUIRE(pipelineFailure.destroy() == Result::SUCCESS);
    TestProgram active(Config());
    REQUIRE(active.create(MTL::PixelFormatRGBA8Unorm, false) == Result::SUCCESS);
    for (U64 i = 0; i < MetalProgramCache::MaxPipelines + 1; ++i) {
        auto variant = Config();
        variant.shaders[DeviceType::Metal][1] = Bytes(prefix + "fragment float4 main0() { return float4(" + std::to_string(i + 2) + ",0,0,1); }");
        TestProgram program(variant);
        REQUIRE(program.create(MTL::PixelFormatRGBA8Unorm, false) == Result::SUCCESS);
        REQUIRE(program.destroy() == Result::SUCCESS);
    }
    RenderActive(device, active);
    REQUIRE(active.destroy() == Result::SUCCESS);
    const auto librariesAfterEviction = counters.libraries;
    const auto pipelinesAfterEviction = counters.pipelines;
    TestProgram recreated(Config());
    REQUIRE(recreated.create(MTL::PixelFormatRGBA8Unorm, false) == Result::SUCCESS);
    CHECK(counters.libraries == librariesAfterEviction + 1);
    CHECK(counters.pipelines == pipelinesAfterEviction + 1);
    RenderActive(device, recreated);
    REQUIRE(recreated.destroy() == Result::SUCCESS);
}
