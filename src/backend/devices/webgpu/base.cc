#include <atomic>
#include <cstring>
#include <memory>
#include <utility>

#include "jetstream/logger.hh"

#include "jetstream/backend/devices/webgpu/base.hh"

namespace Jetstream::Backend {

namespace {

WGPUInstance preinitializedInstance = nullptr;
WGPUAdapter preinitializedAdapter = nullptr;
WGPUDevice preinitializedDevice = nullptr;
std::atomic<bool> initializationCancelled{false};

struct RequestContext {
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    std::function<void(Result)> completion;
};

std::string WebGPUMessage(const WGPUStringView& message) {
    if (!message.data) {
        return "Unknown error";
    }

    const std::size_t length = message.length == WGPU_STRLEN
        ? std::strlen(message.data)
        : message.length;
    return std::string(message.data, length);
}

void ReleaseHandles(WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device) {
    if (device) {
        wgpuDeviceDestroy(device);
        wgpuDeviceRelease(device);
    }
    if (adapter) {
        wgpuAdapterRelease(adapter);
    }
    if (instance) {
        wgpuInstanceRelease(instance);
    }
}

void ReleasePreinitializedHandles() {
    const auto instance = preinitializedInstance;
    const auto adapter = preinitializedAdapter;
    const auto device = preinitializedDevice;
    preinitializedInstance = nullptr;
    preinitializedAdapter = nullptr;
    preinitializedDevice = nullptr;
    ReleaseHandles(instance, adapter, device);
}

void OnDeviceRequest(WGPURequestDeviceStatus status,
                     WGPUDevice device,
                     WGPUStringView message,
                     void* userdata1,
                     void*) {
    std::unique_ptr<RequestContext> context(static_cast<RequestContext*>(userdata1));
    if (status != WGPURequestDeviceStatus_Success || !device) {
        JST_ERROR("[WebGPU] Failed to request device: {}", WebGPUMessage(message));
        ReleaseHandles(context->instance, context->adapter, nullptr);
        context->completion(Result::ERROR);
        return;
    }

    if (initializationCancelled.load(std::memory_order_acquire)) {
        ReleaseHandles(context->instance, context->adapter, device);
        context->completion(Result::ERROR);
        return;
    }

    preinitializedInstance = context->instance;
    preinitializedAdapter = context->adapter;
    preinitializedDevice = device;
    context->completion(Result::SUCCESS);
}

void OnAdapterRequest(WGPURequestAdapterStatus status,
                      WGPUAdapter adapter,
                      WGPUStringView message,
                      void* userdata1,
                      void*) {
    std::unique_ptr<RequestContext> context(static_cast<RequestContext*>(userdata1));
    if (status != WGPURequestAdapterStatus_Success || !adapter) {
        JST_ERROR("[WebGPU] Failed to request adapter: {}", WebGPUMessage(message));
        ReleaseHandles(context->instance, nullptr, nullptr);
        context->completion(Result::ERROR);
        return;
    }

    if (initializationCancelled.load(std::memory_order_acquire)) {
        ReleaseHandles(context->instance, adapter, nullptr);
        context->completion(Result::ERROR);
        return;
    }

    context->adapter = adapter;

    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    descriptor.label = {"CyberEther", WGPU_STRLEN};

    WGPURequestDeviceCallbackInfo callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowSpontaneous;
    callback.callback = OnDeviceRequest;
    callback.userdata1 = context.release();
    (void)wgpuAdapterRequestDevice(adapter, &descriptor, callback);
}

}  // namespace

Result WebGPU::InitializeAsync(std::function<void(Result)> completion) {
    if (!completion) {
        return Result::ERROR;
    }

    initializationCancelled.store(false, std::memory_order_release);

    WGPUInstance instance = wgpuCreateInstance(nullptr);
    if (!instance) {
        JST_ERROR("[WebGPU] Failed to create instance.");
        return Result::ERROR;
    }

    auto context = std::make_unique<RequestContext>();
    context->instance = instance;
    context->completion = std::move(completion);

    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    WGPURequestAdapterCallbackInfo callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowSpontaneous;
    callback.callback = OnAdapterRequest;
    callback.userdata1 = context.release();
    (void)wgpuInstanceRequestAdapter(instance, &options, callback);
    return Result::SUCCESS;
}

void WebGPU::CancelInitialization() {
    initializationCancelled.store(true, std::memory_order_release);
    ReleasePreinitializedHandles();
}

WebGPU::WebGPU(const Config& _config) : config(_config), cache({}) {
    if (!preinitializedInstance || !preinitializedAdapter || !preinitializedDevice) {
        JST_FATAL("WebGPU must be initialized on the browser application thread.");
        throw Result::FATAL;
    }

    instance = preinitializedInstance;
    adapter = preinitializedAdapter;
    device = preinitializedDevice;
    preinitializedInstance = nullptr;
    preinitializedAdapter = nullptr;
    preinitializedDevice = nullptr;

    // Print device information.

    JST_WARN("Due to current Emscripten limitations the device values are inaccurate.");
    JST_INFO("-----------------------------------------------------");
    JST_INFO("Jetstream Heterogeneous Backend [WebGPU]")
    JST_INFO("-----------------------------------------------------");
    JST_INFO("Device Name:     {}", getDeviceName());
    JST_INFO("Device Type:     {}", getPhysicalDeviceType());
    JST_INFO("API Version:     {}", getApiVersion());
    JST_INFO("Unified Memory:  {}", hasUnifiedMemory() ? "YES" : "NO");
    JST_INFO("Processor Count: {}", getTotalProcessorCount());
    JST_INFO("Device Memory:   {:.2f} GB", static_cast<F32>(getPhysicalMemory()) / (1024*1024*1024));
    JST_INFO("-----------------------------------------------------");
}

WebGPU::~WebGPU() {
    if (device) {
        wgpuDeviceDestroy(device);
        wgpuDeviceRelease(device);
    }
    if (adapter) {
        wgpuAdapterRelease(adapter);
    }
    if (instance) {
        wgpuInstanceRelease(instance);
    }
}

std::string WebGPU::getDeviceName() const {
    return cache.deviceName;
}

std::string WebGPU::getApiVersion() const {
    return cache.apiVersion;
}

PhysicalDeviceType WebGPU::getPhysicalDeviceType() const {
    return cache.physicalDeviceType;
}

bool WebGPU::hasUnifiedMemory() const {
    return cache.hasUnifiedMemory;
}

U64 WebGPU::getPhysicalMemory() const {
    return cache.physicalMemory;
}

U64 WebGPU::getTotalProcessorCount() const {
    return cache.totalProcessorCount;
}

bool WebGPU::getLowPowerStatus() const {
    // TODO: Pool power status periodically.
    return cache.lowPowerStatus;
}

U64 WebGPU::getThermalState() const {
    // TODO: Pool thermal state periodically.
    return cache.getThermalState;
}

}  // namespace Jetstream::Backend
