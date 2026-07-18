#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "jetstream/backend/base.hh"
#include "jetstream/detail/instance_remote_supervisor.hh"
#include "jetstream/instance.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/logger.hh"

using namespace Jetstream;

namespace {

struct DeviceNameCase {
    DeviceType device;
    const char* name;
    const char* prettyName;
    const char* mixedCaseName;
};

constexpr std::array<DeviceNameCase, 6> DeviceNameCases = {{
    {DeviceType::None, "none", "None", "NoNe"},
    {DeviceType::CPU, "cpu", "CPU", "CpU"},
    {DeviceType::CUDA, "cuda", "CUDA", "CuDa"},
    {DeviceType::Metal, "metal", "Metal", "MeTaL"},
    {DeviceType::Vulkan, "vulkan", "Vulkan", "VuLkAn"},
    {DeviceType::WebGPU, "webgpu", "WebGPU", "WeBgPu"},
}};

constexpr std::array<DeviceType, 5> ConcreteDeviceTypes = {
    DeviceType::CPU,
    DeviceType::CUDA,
    DeviceType::Metal,
    DeviceType::Vulkan,
    DeviceType::WebGPU,
};

constexpr std::array<DeviceType, 4> InvalidBackendIds = {
    DeviceType::None,
    static_cast<DeviceType>(0),
    DeviceType::CPU | DeviceType::CUDA,
    static_cast<DeviceType>(255),
};

struct CodecNameCase {
    Instance::Remote::CodecType codec;
    const char* name;
    const char* prettyName;
    const char* mixedCaseName;
};

constexpr std::array<CodecNameCase, 4> CodecNameCases = {{
    {Instance::Remote::CodecType::H264, "h264", "H264", "H264"},
    {Instance::Remote::CodecType::AV1, "av1", "AV1", "Av1"},
    {Instance::Remote::CodecType::VP8, "vp8", "VP8", "Vp8"},
    {Instance::Remote::CodecType::VP9, "vp9", "VP9", "vP9"},
}};

struct EncoderNameCase {
    Instance::Remote::EncoderType encoder;
    const char* name;
    const char* prettyName;
    const char* mixedCaseName;
};

constexpr std::array<EncoderNameCase, 6> EncoderNameCases = {{
    {Instance::Remote::EncoderType::Auto, "auto", "Auto", "AuTo"},
    {Instance::Remote::EncoderType::Software, "software", "Software", "SoFtWaRe"},
    {Instance::Remote::EncoderType::NVENC, "nvenc", "NVENC", "NvEnC"},
    {Instance::Remote::EncoderType::V4L2, "v4l2", "V4L2", "V4l2"},
    {Instance::Remote::EncoderType::VideoToolbox,
     "videotoolbox",
     "VideoToolbox",
     "ViDeOtOoLbOx"},
    {Instance::Remote::EncoderType::MediaFoundation,
     "mediafoundation",
     "MediaFoundation",
     "MeDiAfOuNdAtIoN"},
}};

template<typename Callable>
void RequireResultError(Callable&& callable) {
    Result thrown = Result::SUCCESS;
    try {
        callable();
    } catch (const Result result) {
        thrown = result;
    }
    REQUIRE(thrown == Result::ERROR);
}

}  // namespace

TEST_CASE("Startup configuration defaults are stable",
          "[core][application][instance][backend]") {
    const Instance::Config instance;
    REQUIRE_FALSE(instance.device.has_value());
    REQUIRE(instance.deviceId == 0);
    REQUIRE_FALSE(instance.compositor.has_value());
    REQUIRE_FALSE(instance.headless);
    REQUIRE(instance.size.x == 1920);
    REQUIRE(instance.size.y == 1080);
    REQUIRE(instance.scale == 1.0f);
    REQUIRE(instance.framerate == 60);
    REQUIRE(instance.pythonRuntimePath.empty());

    const Backend::Config backend;
    REQUIRE(backend.deviceId == 0);
#ifdef JST_DEBUG_MODE
    REQUIRE(backend.validationEnabled);
#else
    REQUIRE_FALSE(backend.validationEnabled);
#endif
    REQUIRE(backend.multisampling == 4);
    REQUIRE_FALSE(backend.headless);
    REQUIRE(backend.pythonRuntimePath.empty());

    const Instance::Remote::Config remote;
    REQUIRE(remote.broker == "https://cyberether.org");
    REQUIRE_FALSE(remote.autoJoinSessions);
    REQUIRE(remote.framerate == 30);
    REQUIRE(remote.encoder == Instance::Remote::EncoderType::Auto);
    REQUIRE(remote.codec == Instance::Remote::CodecType::H264);
}

TEST_CASE("Application configuration snapshots retain every explicit value",
          "[core][application][instance][backend][configuration]") {
    Instance::Config instance = {
        .device = DeviceType::WebGPU,
        .deviceId = 42,
        .compositor = CompositorType::DEFAULT,
        .headless = true,
        .size = {321, 654},
        .scale = 1.75f,
        .framerate = 144,
        .pythonRuntimePath = "instance-python",
    };
    const auto instanceSnapshot = instance;
    instance.device.reset();
    instance.deviceId = 0;
    instance.compositor.reset();
    instance.headless = false;
    instance.size = {1, 1};
    instance.scale = 1.0f;
    instance.framerate = 1;
    instance.pythonRuntimePath.clear();

    REQUIRE(instanceSnapshot.device == DeviceType::WebGPU);
    REQUIRE(instanceSnapshot.deviceId == 42);
    REQUIRE(instanceSnapshot.compositor == CompositorType::DEFAULT);
    REQUIRE(instanceSnapshot.headless);
    REQUIRE(instanceSnapshot.size.x == 321);
    REQUIRE(instanceSnapshot.size.y == 654);
    REQUIRE(instanceSnapshot.scale == 1.75f);
    REQUIRE(instanceSnapshot.framerate == 144);
    REQUIRE(instanceSnapshot.pythonRuntimePath == "instance-python");

    Backend::Config backend = {
        .deviceId = 17,
        .validationEnabled = true,
        .multisampling = 8,
        .headless = true,
        .pythonRuntimePath = "backend-python",
    };
    const auto backendSnapshot = backend;
    backend.deviceId = 0;
    backend.validationEnabled = false;
    backend.multisampling = 1;
    backend.headless = false;
    backend.pythonRuntimePath.clear();

    REQUIRE(backendSnapshot.deviceId == 17);
    REQUIRE(backendSnapshot.validationEnabled);
    REQUIRE(backendSnapshot.multisampling == 8);
    REQUIRE(backendSnapshot.headless);
    REQUIRE(backendSnapshot.pythonRuntimePath == "backend-python");

    Instance::Remote::Config remote = {
        .broker = "http://remote.invalid/root/",
        .autoJoinSessions = true,
        .framerate = 24,
        .encoder = Instance::Remote::EncoderType::Software,
        .codec = Instance::Remote::CodecType::VP9,
    };
    const auto remoteSnapshot = remote;
    remote.broker.clear();
    remote.autoJoinSessions = false;
    remote.framerate = 1;
    remote.encoder = Instance::Remote::EncoderType::Auto;
    remote.codec = Instance::Remote::CodecType::H264;

    REQUIRE(remoteSnapshot.broker == "http://remote.invalid/root/");
    REQUIRE(remoteSnapshot.autoJoinSessions);
    REQUIRE(remoteSnapshot.framerate == 24);
    REQUIRE(remoteSnapshot.encoder == Instance::Remote::EncoderType::Software);
    REQUIRE(remoteSnapshot.codec == Instance::Remote::CodecType::VP9);
}

// TODO: Expose the native settings-to-Instance and settings-to-Remote builders
// as pure detail helpers. They are currently anonymous in run_native.cc, and
// reaching them through Run continues into real application initialization.

TEST_CASE("Device backend names round-trip without backend initialization",
          "[core][application][backend][conversion]") {
    for (const auto& entry : DeviceNameCases) {
        REQUIRE(std::string(GetDeviceName(entry.device)) == entry.name);
        REQUIRE(std::string(GetDevicePrettyName(entry.device)) == entry.prettyName);
        REQUIRE(StringToDevice(entry.name) == entry.device);
        REQUIRE(StringToDevice(entry.prettyName) == entry.device);
        REQUIRE(StringToDevice(entry.mixedCaseName) == entry.device);

        std::ostringstream stream;
        stream << entry.device;
        REQUIRE(stream.str() == entry.prettyName);
    }

    const std::array<std::string, 6> invalidNames = {
        "",
        "  cpu  ",
        "cpu ",
        "cpu0",
        "not-a-device",
        std::string{"cpu\0suffix", 10},
    };
    for (const auto& name : invalidNames) {
        REQUIRE(StringToDevice(name) == DeviceType::None);
    }

    const std::array<DeviceType, 3> invalidDevices = {
        static_cast<DeviceType>(0),
        DeviceType::CPU | DeviceType::CUDA,
        static_cast<DeviceType>(255),
    };
    for (const auto invalid : invalidDevices) {
        REQUIRE_THROWS_AS(GetDeviceName(invalid), std::out_of_range);
        REQUIRE(std::string(GetDevicePrettyName(invalid)) == "None");

        std::ostringstream stream;
        stream << invalid;
        REQUIRE(stream.str() == "None");
        REQUIRE_FALSE(stream.fail());
    }
}

TEST_CASE("Device flags compose without selecting or initializing a backend",
          "[core][application][backend][enum]") {
    constexpr auto computeDevices = DeviceType::CPU | DeviceType::CUDA;
    constexpr auto graphicsDevices = DeviceType::Metal | DeviceType::Vulkan | DeviceType::WebGPU;

    REQUIRE((computeDevices & DeviceType::CPU) == DeviceType::CPU);
    REQUIRE((computeDevices & DeviceType::CUDA) == DeviceType::CUDA);
    REQUIRE(static_cast<std::uint8_t>(computeDevices & DeviceType::Metal) == 0);
    REQUIRE((graphicsDevices & DeviceType::Metal) == DeviceType::Metal);
    REQUIRE((graphicsDevices & DeviceType::Vulkan) == DeviceType::Vulkan);
    REQUIRE((graphicsDevices & DeviceType::WebGPU) == DeviceType::WebGPU);
}

TEST_CASE("Physical device type formatting covers every public value",
          "[core][application][backend][conversion]") {
    const std::array<std::pair<Backend::PhysicalDeviceType, const char*>, 4> cases = {{
        {Backend::PhysicalDeviceType::UNKNOWN, "UNKNOWN"},
        {Backend::PhysicalDeviceType::DISCRETE, "DISCRETE"},
        {Backend::PhysicalDeviceType::INTEGRATED, "INTEGRATED"},
        {Backend::PhysicalDeviceType::OTHER, "OTHER"},
    }};

    for (const auto& [type, expected] : cases) {
        std::ostringstream stream;
        stream << type;
        REQUIRE(stream.str() == expected);
        REQUIRE_FALSE(stream.fail());
    }

    for (const auto value : {4, 255}) {
        std::ostringstream invalid;
        invalid << static_cast<Backend::PhysicalDeviceType>(value);
        REQUIRE(invalid.fail());
        REQUIRE(invalid.str().empty());
    }
}

TEST_CASE("Backend manager treats configuration as distinct from initialization",
          "[core][application][backend][lifecycle]") {
    Backend::Instance backend;
    Backend::Config config;
    config.deviceId = 7;
    config.headless = true;
    config.pythonRuntimePath = "configured-runtime";

    for (const auto device : ConcreteDeviceTypes) {
        REQUIRE_FALSE(backend.initialized(device));
        REQUIRE(backend.configure(device, config) == Result::SUCCESS);
        REQUIRE_FALSE(backend.initialized(device));
        REQUIRE(backend.configure(device, config) == Result::SUCCESS);
        REQUIRE_FALSE(backend.initialized(device));
    }

    for (const auto device : ConcreteDeviceTypes) {
        REQUIRE(backend.destroy(device) == Result::SUCCESS);
        REQUIRE_FALSE(backend.initialized(device));
        REQUIRE(backend.destroy(device) == Result::SUCCESS);
    }

    REQUIRE(backend.destroyAll() == Result::SUCCESS);
    for (const auto device : ConcreteDeviceTypes) {
        REQUIRE_FALSE(backend.initialized(device));
    }
    REQUIRE(backend.destroyAll() == Result::SUCCESS);
}

TEST_CASE("Backend manager rejects identifiers that cannot name one backend",
          "[core][application][backend][validation]") {
    Backend::Instance backend;
    Backend::Config config;

    for (const auto id : InvalidBackendIds) {
        // Current defect: configure accepts sentinel, combined, and out-of-range
        // identifiers even though no typed backend state can use those keys.
        CHECK(backend.configure(id, config) == Result::ERROR);
        CHECK_FALSE(backend.initialized(id));
    }

    REQUIRE(backend.destroyAll() == Result::SUCCESS);
}

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE

TEST_CASE("CPU backend configuration survives lazy initialization and explicit destroy",
          "[core][application][backend][lifecycle]") {
    Backend::Instance backend;
    Backend::Config configured;
    configured.pythonRuntimePath = "configured-runtime";

    REQUIRE(backend.configure(DeviceType::CPU, configured) == Result::SUCCESS);

    Backend::Config replacementBeforeInitialization;
    replacementBeforeInitialization.headless = true;
    replacementBeforeInitialization.pythonRuntimePath = "replacement-before-initialization";
    REQUIRE(backend.configure(DeviceType::CPU, replacementBeforeInitialization) == Result::SUCCESS);

    const auto* firstState = backend.state<DeviceType::CPU>().get();
    REQUIRE(firstState != nullptr);
    REQUIRE(firstState->getPythonRuntimePath() == "replacement-before-initialization");
    REQUIRE(backend.initialized(DeviceType::CPU));

    Backend::Config replacementAfterInitialization;
    replacementAfterInitialization.pythonRuntimePath = "replacement-after-initialization";
    REQUIRE(backend.configure(DeviceType::CPU, replacementAfterInitialization) == Result::SUCCESS);
    REQUIRE(backend.initialize<DeviceType::CPU>(replacementAfterInitialization) == Result::SUCCESS);
    const auto* repeatedState = backend.state<DeviceType::CPU>().get();
    REQUIRE(repeatedState == firstState);
    REQUIRE(backend.state<DeviceType::CPU>().get() == repeatedState);
    REQUIRE(backend.state<DeviceType::CPU>()->getPythonRuntimePath() ==
            "replacement-before-initialization");

    REQUIRE(backend.destroy(DeviceType::CPU) == Result::SUCCESS);
    REQUIRE_FALSE(backend.initialized(DeviceType::CPU));
    REQUIRE(backend.destroy(DeviceType::CPU) == Result::SUCCESS);
    REQUIRE_FALSE(backend.initialized(DeviceType::CPU));

    const auto& restoredState = backend.state<DeviceType::CPU>();
    REQUIRE(restoredState != nullptr);
    REQUIRE(restoredState->getPythonRuntimePath() == "replacement-before-initialization");

    REQUIRE(backend.destroy(DeviceType::CPU) == Result::SUCCESS);
    Backend::Config replacementAfterDestroy;
    replacementAfterDestroy.pythonRuntimePath = "replacement-after-destroy";
    REQUIRE(backend.configure(DeviceType::CPU, replacementAfterDestroy) == Result::SUCCESS);
    REQUIRE(backend.state<DeviceType::CPU>()->getPythonRuntimePath() ==
            "replacement-after-destroy");

    REQUIRE(backend.destroyAll() == Result::SUCCESS);
    REQUIRE_FALSE(backend.initialized(DeviceType::CPU));

    // Backend state has no configuration snapshot; the empty runtime path is
    // the observable part of the implicit headless lazy-state fallback.
    REQUIRE(backend.state<DeviceType::CPU>()->getPythonRuntimePath().empty());
    REQUIRE(backend.destroyAll() == Result::SUCCESS);
}

#endif

// TODO: Cover GPU backend initialization, failed construction cleanup, and
// teardown through injectable device factories. Direct construction here would
// probe hardware and drivers.

TEST_CASE("Instance guards resources before create and after rejected destroy",
          "[core][application][instance][lifecycle]") {
    auto instance = std::make_shared<Instance>();

    const auto requireInactiveGuards = [&] {
        REQUIRE_FALSE(instance->computing());
        REQUIRE_FALSE(instance->presenting());
        REQUIRE_FALSE(instance->polling());
        REQUIRE(instance->remote() == nullptr);

        bool callbackCalled = false;
        REQUIRE(instance->start() == Result::ERROR);
        REQUIRE(instance->stop() == Result::ERROR);
        REQUIRE(instance->compute() == Result::ERROR);
        REQUIRE(instance->present([&] {
            callbackCalled = true;
            return Result::SUCCESS;
        }) == Result::ERROR);
        REQUIRE_FALSE(callbackCalled);
        REQUIRE(instance->poll(false) == Result::ERROR);
        REQUIRE(instance->poll() == Result::ERROR);

        auto untouchedFlowgraph = std::make_shared<Flowgraph>();
        auto flowgraphOutput = untouchedFlowgraph;
        REQUIRE(instance->flowgraphCreate("graph", {}, flowgraphOutput) == Result::ERROR);
        REQUIRE(flowgraphOutput == untouchedFlowgraph);
        REQUIRE(instance->flowgraphDestroy("graph") == Result::ERROR);

        std::unordered_map<std::string, std::shared_ptr<Flowgraph>> flowgraphs = {
            {"untouched", untouchedFlowgraph},
        };
        REQUIRE(instance->flowgraphList(flowgraphs) == Result::ERROR);
        REQUIRE(flowgraphs.size() == 1);
        REQUIRE(flowgraphs.at("untouched") == untouchedFlowgraph);

        std::shared_ptr<Compositor> compositor;
        std::shared_ptr<Viewport::Generic> viewport;
        std::shared_ptr<Render::Window> render;
        REQUIRE(instance->compositorGet(compositor) == Result::ERROR);
        REQUIRE(instance->viewportGet(viewport) == Result::ERROR);
        REQUIRE(instance->renderGet(render) == Result::ERROR);
        REQUIRE(compositor == nullptr);
        REQUIRE(viewport == nullptr);
        REQUIRE(render == nullptr);
    };

    requireInactiveGuards();
    REQUIRE(instance->destroy() == Result::ERROR);
    requireInactiveGuards();
    REQUIRE(instance->destroy() == Result::ERROR);
    requireInactiveGuards();
}

// TODO: Inject backend, viewport, render, compositor, and font-resource factories
// into Instance. They are needed to test successful create/start/stop/destroy,
// guards after successful destroy, and rollback at each create/stop teardown
// boundary without opening a window or requiring a GPU.
// TODO: The render factory must expose scripted begin/end/cancel results so
// presentation can cover SKIP, callback/compositor/flowgraph errors, cancellation,
// remote capture, and presenting-state transitions without a real frame.
// TODO: Inject a Flowgraph factory into Instance to verify ownership, duplicate
// names, missing names, list snapshots, start-on-insert, and create/start/destroy
// rollback independently of concrete render and scheduler resources.

TEST_CASE("Remote codec names round-trip and reject unknown values",
          "[core][application][instance][remote][conversion]") {
    REQUIRE(RemoteCodecTypes.size() == CodecNameCases.size());

    for (std::size_t index = 0; index < CodecNameCases.size(); ++index) {
        const auto& entry = CodecNameCases[index];
        REQUIRE(RemoteCodecTypes[index] == entry.codec);
        REQUIRE(GetRemoteCodecName(entry.codec) == entry.name);
        REQUIRE(std::string(GetRemoteCodecPrettyName(entry.codec)) == entry.prettyName);
        REQUIRE(StringToRemoteCodec(entry.name) == entry.codec);
        REQUIRE(StringToRemoteCodec(entry.prettyName) == entry.codec);
        REQUIRE(StringToRemoteCodec(entry.mixedCaseName) == entry.codec);

        std::ostringstream stream;
        stream << entry.codec;
        REQUIRE(stream.str() == entry.prettyName);
    }

    RequireResultError([] { (void)StringToRemoteCodec(""); });
    RequireResultError([] { (void)StringToRemoteCodec(" h264 "); });
    RequireResultError([] { (void)StringToRemoteCodec("h264 "); });
    RequireResultError([] { (void)StringToRemoteCodec("h-264"); });
    RequireResultError([] { (void)StringToRemoteCodec("vp10"); });
    RequireResultError([] { (void)StringToRemoteCodec(std::string{"h264\0suffix", 11}); });
    RequireResultError([] { (void)StringToRemoteCodec("not-a-codec"); });

    for (const auto value : {4, 255}) {
        const auto invalid = static_cast<Instance::Remote::CodecType>(value);
        RequireResultError([invalid] { (void)GetRemoteCodecName(invalid); });
        REQUIRE(std::string(GetRemoteCodecPrettyName(invalid)) == "Unknown");

        std::ostringstream stream;
        stream << invalid;
        REQUIRE(stream.str() == "Unknown");
        REQUIRE_FALSE(stream.fail());
    }
}

TEST_CASE("Remote encoder names round-trip and reject unknown values",
          "[core][application][instance][remote][conversion]") {
    REQUIRE(RemoteEncoderTypes.size() == EncoderNameCases.size());

    for (std::size_t index = 0; index < EncoderNameCases.size(); ++index) {
        const auto& entry = EncoderNameCases[index];
        REQUIRE(RemoteEncoderTypes[index] == entry.encoder);
        REQUIRE(GetRemoteEncoderName(entry.encoder) == entry.name);
        REQUIRE(std::string(GetRemoteEncoderPrettyName(entry.encoder)) == entry.prettyName);
        REQUIRE(StringToRemoteEncoder(entry.name) == entry.encoder);
        REQUIRE(StringToRemoteEncoder(entry.prettyName) == entry.encoder);
        REQUIRE(StringToRemoteEncoder(entry.mixedCaseName) == entry.encoder);

        std::ostringstream stream;
        stream << entry.encoder;
        REQUIRE(stream.str() == entry.prettyName);
    }

    RequireResultError([] { (void)StringToRemoteEncoder(""); });
    RequireResultError([] { (void)StringToRemoteEncoder(" auto "); });
    RequireResultError([] { (void)StringToRemoteEncoder("auto "); });
    RequireResultError([] { (void)StringToRemoteEncoder("video-toolbox"); });
    RequireResultError([] { (void)StringToRemoteEncoder("nvenc0"); });
    RequireResultError([] { (void)StringToRemoteEncoder(std::string{"auto\0suffix", 11}); });
    RequireResultError([] { (void)StringToRemoteEncoder("not-an-encoder"); });

    for (const auto value : {6, 255}) {
        const auto invalid = static_cast<Instance::Remote::EncoderType>(value);
        RequireResultError([invalid] { (void)GetRemoteEncoderName(invalid); });
        REQUIRE(std::string(GetRemoteEncoderPrettyName(invalid)) == "Unknown");

        std::ostringstream stream;
        stream << invalid;
        REQUIRE(stream.str() == "Unknown");
        REQUIRE_FALSE(stream.fail());
    }
}

TEST_CASE("Remote without a viewport is inert and reports unsupported operations",
          "[core][application][instance][remote]") {
    Instance::Remote remote(nullptr);

    const auto requireInertState = [&] {
        REQUIRE_FALSE(remote.supported());
        REQUIRE_FALSE(remote.started());
        for (const auto codec : RemoteCodecTypes) {
            REQUIRE(remote.available(codec).empty());
        }
        REQUIRE(remote.available(static_cast<Instance::Remote::CodecType>(255)).empty());
        REQUIRE(remote.roomId().empty());
        REQUIRE(remote.accessToken().empty());
        REQUIRE(remote.inviteUrl().empty());
        REQUIRE(remote.clients().empty());
        REQUIRE(remote.waitlist().empty());
    };

    requireInertState();

    REQUIRE(remote.captureFrame() == Result::SUCCESS);
    REQUIRE(remote.captureFrame() == Result::SUCCESS);
    REQUIRE(remote.approveClient("") == Result::ERROR);
    REQUIRE(remote.approveClient("client") == Result::ERROR);
    REQUIRE(remote.create({}) == Result::ERROR);
    REQUIRE(remote.create({}) == Result::ERROR);

    Instance::Remote::Config invalidConfig;
    invalidConfig.broker.clear();
    invalidConfig.framerate = 0;
    invalidConfig.encoder = static_cast<Instance::Remote::EncoderType>(255);
    invalidConfig.codec = static_cast<Instance::Remote::CodecType>(255);
    REQUIRE(remote.create(invalidConfig) == Result::ERROR);
    requireInertState();

    REQUIRE(remote.destroy() == Result::SUCCESS);
    REQUIRE(remote.destroy() == Result::SUCCESS);
    requireInertState();

    REQUIRE(remote.create({}) == Result::ERROR);
    REQUIRE(remote.destroy() == Result::SUCCESS);
    requireInertState();
}

TEST_CASE("Remote broker scheme validation matches the transport",
          "[core][application][instance][remote]") {
    REQUIRE(IsRemoteBrokerSchemeSupported("https://example.com"));
    REQUIRE(IsRemoteBrokerSchemeSupported("http://localhost:8080/root"));
    REQUIRE(IsRemoteBrokerSchemeSupported("https://example.com/path?key=value"));
    REQUIRE_FALSE(IsRemoteBrokerSchemeSupported("ftp://example.com"));
    REQUIRE_FALSE(IsRemoteBrokerSchemeSupported("wss://example.com"));
    REQUIRE_FALSE(IsRemoteBrokerSchemeSupported("HTTPS://example.com"));
    REQUIRE_FALSE(IsRemoteBrokerSchemeSupported("https:/example.com"));
    REQUIRE_FALSE(IsRemoteBrokerSchemeSupported(" https://example.com"));
    REQUIRE_FALSE(IsRemoteBrokerSchemeSupported(""));

    Instance::Remote remote(nullptr);
    Instance::Remote::Config config;
    config.broker = "ftp://example.com";
    JST_LOG_LAST_ERROR().clear();
    REQUIRE(remote.create(config) == Result::ERROR);
    REQUIRE(JST_LOG_LAST_ERROR() == "[REMOTE] Broker URL must use HTTP or HTTPS.");
}

TEST_CASE("Remote supervisor treats a null remote as a stopped transport",
          "[core][application][instance][remote][supervisor]") {
    for (const bool autoJoin : {false, true}) {
        Instance::Remote::Supervisor supervisor(nullptr, autoJoin);

        REQUIRE_NOTHROW(supervisor.stop());
        REQUIRE_NOTHROW(supervisor.start());
        REQUIRE_NOTHROW(supervisor.start());
        REQUIRE_NOTHROW(supervisor.stop());
        REQUIRE_NOTHROW(supervisor.stop());
    }
}

// TODO: Add injectable frame-capture, media-pipeline, and signaller transport
// factories for supported Remote create/destroy and rollback at stream, broker,
// room, capture, and submission-thread boundaries. The concrete path requires
// Vulkan, GStreamer, threads, and network services.
// TODO: Give Supervisor an injectable remote transport and wait strategy so
// waitlist deduplication, short IDs, authorization-code normalization, auto-join,
// rejected approval, and stop races can be tested without sleeping or reading stdin.
// Its print path also needs a nullable session view before null transport output
// can be exercised without dereferencing the remote pointer.
// TODO: Put signaller and data-channel protocol handling behind an injectable
// transport/input sink. This is needed for malformed/oversized messages, session
// ownership, authorization sends, SDP/ICE transitions, and mouse, wheel, and
// keyboard translation without WebRTC or a real viewport.
// TODO: Extract broker endpoint transformation into a pure helper and inject the
// broker transport so HTTP/HTTPS scheme conversion, trailing-slash normalization,
// invite construction, and connect/create-room/start-stream rollback are testable
// without opening a socket.
