#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <jetstream/tools/numeric.hh>
#include <jetstream/memory/axis.hh>

#include "miniaudio.h"

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#endif

namespace Jetstream::Modules {

namespace {

bool DeriveBufferSizes(const U64 inputFrameCount,
                       const U32 channelCount,
                       const U32 inputSampleRate,
                       const U32 outputSampleRate,
                       U64& outputSize,
                       U64& circularBufferSize) {
    outputSize = 0;
    circularBufferSize = 0;
    if (inputSampleRate == 0 || outputSampleRate == 0) {
        return false;
    }

    U64 outputWhole = 0;
    U64 outputRemainder = 0;
    if (!detail::CheckedMultiply(inputFrameCount / inputSampleRate,
                                 outputSampleRate,
                                 outputWhole) ||
        !detail::CheckedMultiply(inputFrameCount % inputSampleRate,
                                 outputSampleRate,
                                 outputRemainder)) {
        return false;
    }

    const U64 roundedRemainder = outputRemainder == 0 ?
        0 : 1 + (outputRemainder - 1) / inputSampleRate;
    U64 candidateOutputFrameCount = 0;
    U64 bufferedInputFrameCount = 0;
    if (!detail::CheckedAdd(outputWhole,
                            roundedRemainder,
                            candidateOutputFrameCount) ||
        !detail::CheckedMultiply(inputFrameCount, 20,
                                 bufferedInputFrameCount)) {
        return false;
    }

    const U64 circularBufferFrameCount =
        std::max(bufferedInputFrameCount, candidateOutputFrameCount);
    return detail::CheckedMultiply(candidateOutputFrameCount,
                                   channelCount,
                                   outputSize) &&
           detail::CheckedMultiply(circularBufferFrameCount,
                                   channelCount,
                                   circularBufferSize);
}

}  // namespace

AudioImpl::AudioImpl() = default;
AudioImpl::~AudioImpl() = default;

struct AudioImpl::Impl {
    ma_device_config deviceConfig;
    ma_device deviceCtx;
    ma_resampler_config resamplerConfig;
    ma_resampler resamplerCtx;
    bool resamplerInitialized = false;

    static void callback(ma_device* pDevice, void* pOutput, const void* pInput,
                         ma_uint32 frameCount);
    static std::vector<std::pair<ma_device_id, std::string>> GetAvailableDevices();
    static void GenerateUniqueName(std::string& name, const ma_device_id& id);
};

void AudioImpl::Impl::GenerateUniqueName(std::string& name, const ma_device_id& id) {
    if (id.pulse[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.pulse));
    } else if (id.alsa[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.alsa));
    } else if (id.jack != 0) {
        name = jst::fmt::format("{} ({})", name, id.jack);
    } else if (id.coreaudio[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.coreaudio));
    } else if (id.sndio[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.sndio));
    } else if (id.audio4[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.audio4));
    } else if (id.oss[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.oss));
    } else if (id.aaudio != 0) {
        name = jst::fmt::format("{} ({})", name, id.aaudio);
    } else if (id.opensl != 0) {
        name = jst::fmt::format("{} ({})", name, id.opensl);
    } else if (id.webaudio[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.webaudio));
    } else if (id.custom.i != 0) {
        name = jst::fmt::format("{} ({})", name, id.custom.i);
    } else if (id.nullbackend != 0) {
        name = jst::fmt::format("{} ({})", name, id.nullbackend);
    } else if (id.winmm != 0) {
        name = jst::fmt::format("{} ({})", name, id.winmm);
    } else if (id.wasapi[0] != '\0') {
        const U64 sum = std::accumulate(id.wasapi, id.wasapi + sizeof(id.wasapi), 0ULL);
        name = jst::fmt::format("{} ({:08X})", name, sum);
    } else if (id.dsound[0] != '\0') {
        const U64 sum = std::accumulate(id.dsound, id.dsound + sizeof(id.dsound), 0ULL);
        name = jst::fmt::format("{} ({:08X})", name, sum);
    }
}

std::vector<std::pair<ma_device_id, std::string>> AudioImpl::Impl::GetAvailableDevices() {
    std::vector<std::pair<ma_device_id, std::string>> devices;

    devices.push_back({{}, "Default"});

    ma_context context;

    if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to initialize audio context.");
        return devices;
    }

    ma_device_info* pPlaybackDeviceInfos;
    ma_uint32 playbackDeviceCount;

    if (ma_context_get_devices(&context, &pPlaybackDeviceInfos, &playbackDeviceCount,
                               nullptr, nullptr) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to retrieve audio devices.");
        ma_context_uninit(&context);
        return devices;
    }

    std::unordered_map<std::string, U64> nameCount;

    for (ma_uint32 i = 0; i < playbackDeviceCount; i++) {
        nameCount[pPlaybackDeviceInfos[i].name] = 0;
    }

    for (ma_uint32 i = 0; i < playbackDeviceCount; i++) {
        nameCount[pPlaybackDeviceInfos[i].name] += 1;
    }

    for (ma_uint32 i = 0; i < playbackDeviceCount; i++) {
        const auto& id = pPlaybackDeviceInfos[i].id;
        std::string name = pPlaybackDeviceInfos[i].name;

        if (nameCount.at(name) > 1) {
            Impl::GenerateUniqueName(name, id);
        }

        devices.push_back({id, name});
    }

    ma_context_uninit(&context);

    return devices;
}

void AudioImpl::Impl::callback(ma_device* pDevice, void* pOutput, const void*,
                               ma_uint32 frameCount) {
    auto* audioCircularBuffer = reinterpret_cast<Tools::CircularBuffer<F32>*>(pDevice->pUserData);
    const U64 sampleCount = static_cast<U64>(frameCount) *
                            pDevice->playback.channels;

    if (audioCircularBuffer->size() >= sampleCount) {
        (void)audioCircularBuffer->pop(
            reinterpret_cast<F32*>(pOutput), sampleCount);
    }
}

AudioImpl::DeviceList AudioImpl::ListAvailableDevices() {
    const auto& devices = Impl::GetAvailableDevices();

    DeviceList deviceList;
    for (const auto& [_, name] : devices) {
        deviceList.push_back(name);
    }

    return deviceList;
}

Result AudioImpl::validate() {
    const auto& config = *candidate();
    validatedInSampleRate = 0;
    validatedOutSampleRate = 0;
    validatedOutputSize = 0;
    validatedOutputSizeBytes = 0;
    validatedCircularBufferSize = 0;
    validatedCircularBufferSizeBytes = 0;
    validatedSampleAxis = 0;
    validatedBatchAxis.reset();
    validatedChannelAxis.reset();
    validatedChannelCount = 1;

    constexpr F64 maxSampleRate =
        static_cast<F64>(std::numeric_limits<U32>::max());

    if (!std::isfinite(config.inSampleRate) ||
        config.inSampleRate < 1.0f ||
        static_cast<F64>(config.inSampleRate) > maxSampleRate) {
        JST_ERROR("[MODULE_AUDIO] Input sample rate must be finite and within "
                  "the U32 range.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.outSampleRate) ||
        config.outSampleRate < 1.0f ||
        static_cast<F64>(config.outSampleRate) > maxSampleRate) {
        JST_ERROR("[MODULE_AUDIO] Output sample rate must be finite and within "
                  "the U32 range.");
        return Result::ERROR;
    }

    const U32 candidateInSampleRate = static_cast<U32>(config.inSampleRate);
    const U32 candidateOutSampleRate = static_cast<U32>(config.outSampleRate);

    if (!inputs().contains("buffer")) {
        validatedInSampleRate = candidateInSampleRate;
        validatedOutSampleRate = candidateOutSampleRate;
        return Result::SUCCESS;
    }

    const Tensor& inputBuffer = inputs().at("buffer").tensor;
    if (!inputBuffer.validShape() || inputBuffer.size() == 0) {
        validatedInSampleRate = candidateInSampleRate;
        validatedOutSampleRate = candidateOutSampleRate;
        return Result::SUCCESS;
    }

    SignalAxes axes;
    if (ResolveSignalAxes(inputBuffer, axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }
    const U64 candidateChannelCount = axes.channel ?
        inputBuffer.shape(*axes.channel) : 1;
    if (candidateChannelCount != 1 && candidateChannelCount != 2) {
        JST_ERROR("[MODULE_AUDIO] Input must contain one or two audio channels.");
        return Result::ERROR;
    }
    const Index expectedRank = 1 + static_cast<Index>(axes.batch.has_value()) +
                               static_cast<Index>(axes.channel.has_value());
    if (inputBuffer.rank() != expectedRank) {
        JST_ERROR("[MODULE_AUDIO] Input must contain only a sample axis and "
                  "optional batch and channel axes.");
        return Result::ERROR;
    }

    const U32 candidateChannelCountU32 =
        static_cast<U32>(candidateChannelCount);
    const U64 inputFrameCount = inputBuffer.size() / candidateChannelCount;
    U64 outputSize = 0;
    U64 circularBufferSize = 0;
    if (!DeriveBufferSizes(inputFrameCount,
                           candidateChannelCountU32,
                           candidateInSampleRate,
                           candidateOutSampleRate,
                           outputSize,
                           circularBufferSize)) {
        JST_ERROR("[MODULE_AUDIO] Output or circular buffer size exceeds "
                  "the supported range.");
        return Result::ERROR;
    }

    U64 outputSizeBytes = 0;
    U64 circularBufferSizeBytes = 0;
    if (!detail::CheckedMultiply(outputSize,
                                 static_cast<U64>(sizeof(F32)),
                                 outputSizeBytes) ||
        !detail::CheckedMultiply(circularBufferSize,
                                 static_cast<U64>(sizeof(F32)),
                                 circularBufferSizeBytes)) {
        JST_ERROR("[MODULE_AUDIO] Output or circular buffer layout exceeds "
                  "the supported range.");
        return Result::ERROR;
    }

    validatedInSampleRate = candidateInSampleRate;
    validatedOutSampleRate = candidateOutSampleRate;
    validatedOutputSize = outputSize;
    validatedOutputSizeBytes = outputSizeBytes;
    validatedCircularBufferSize = circularBufferSize;
    validatedCircularBufferSizeBytes = circularBufferSizeBytes;
    validatedSampleAxis = *axes.sample;
    validatedBatchAxis = axes.batch;
    validatedChannelAxis = axes.channel;
    validatedChannelCount = candidateChannelCountU32;

    return Result::SUCCESS;
}

Result AudioImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::BROWSER_MAIN_THREAD));

    JST_CHECK(defineInterfaceInput("buffer"));

    return Result::SUCCESS;
}

Result AudioImpl::create() {
    pimpl = std::make_unique<Impl>();
    sampleAxis = validatedSampleAxis;
    batchAxis = validatedBatchAxis;
    channelAxis = validatedChannelAxis;
    channelCount = validatedChannelCount;

    // Configure audio resampler.

    pimpl->resamplerConfig = ma_resampler_config_init(
        ma_format_f32,
        channelCount,
        validatedInSampleRate,
        validatedOutSampleRate,
        ma_resample_algorithm_linear
    );
    pimpl->resamplerConfig.linear.lpfOrder = 8;

    if (ma_resampler_init(&pimpl->resamplerConfig, nullptr, &pimpl->resamplerCtx) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to create audio resampler.");
        return Result::ERROR;
    }
    pimpl->resamplerInitialized = true;

    // Get available audio devices.

    const auto& devices = Impl::GetAvailableDevices();

    if (devices.empty()) {
        JST_ERROR("[MODULE_AUDIO] No audio devices found.");
        return Result::INCOMPLETE;
    }

    ma_device_id selectedDeviceId;
    bool foundConfigDevice = false;
    bool useDefaultDevice = deviceName == "Default" ||
                            deviceName == "default" ||
                            deviceName.empty();

    JST_DEBUG("[MODULE_AUDIO] Found audio devices:");
    for (U64 i = 0; i < devices.size(); i++) {
        const auto& id = devices[i].first;
        std::string name = devices[i].second;

        if (name == deviceName) {
            selectedDeviceId = id;
            foundConfigDevice = true;
        }

        JST_DEBUG("[MODULE_AUDIO]   [{}]: {}", i, name);
    }

    if (!foundConfigDevice && !useDefaultDevice) {
        JST_WARN("[MODULE_AUDIO] Device '{}' not found, using default.", deviceName);
    }

    // Configure audio device.

    pimpl->deviceConfig = ma_device_config_init(ma_device_type_playback);
    pimpl->deviceConfig.playback.pDeviceID = (!foundConfigDevice || useDefaultDevice) ?
                                              nullptr : &selectedDeviceId;
    pimpl->deviceConfig.playback.format = ma_format_f32;
    pimpl->deviceConfig.playback.channels = channelCount;
    pimpl->deviceConfig.sampleRate = validatedOutSampleRate;
    pimpl->deviceConfig.dataCallback = Impl::callback;
    pimpl->deviceConfig.pUserData = &circularBuffer;

    if (ma_device_init(nullptr, &pimpl->deviceConfig, &pimpl->deviceCtx) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to open audio device.");
        ma_resampler_uninit(&pimpl->resamplerCtx, nullptr);
        pimpl->resamplerInitialized = false;
        return Result::INCOMPLETE;
    }

    resolvedDeviceName = pimpl->deviceCtx.playback.name;

    if (ma_device_start(&pimpl->deviceCtx) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to start playback device.");
        ma_device_uninit(&pimpl->deviceCtx);
        ma_resampler_uninit(&pimpl->resamplerCtx, nullptr);
        pimpl->resamplerInitialized = false;
        return Result::ERROR;
    }

    // Set initial volume.
    ma_device_set_master_volume(&pimpl->deviceCtx, volume);

    // Allocate resampler scratch buffer.

    JST_CHECK(buffer.create(device(), DataType::F32, {validatedOutputSize}));
    const Tensor& input = inputs().at("buffer").tensor;
    const U64 batchStride = input.shape(sampleAxis) * channelCount;
    gatherInput = input.stride(sampleAxis) != channelCount ||
                  (channelAxis && input.stride(*channelAxis) != 1) ||
                  (batchAxis && input.stride(*batchAxis) != batchStride);
    if (gatherInput) {
        orderedInput.resize(input.size());
    } else {
        orderedInput.clear();
    }
    pendingInput.clear();

    // Initialize circular buffer.

    JST_CHECK(circularBuffer.resize(validatedCircularBufferSize));

    return Result::SUCCESS;
}

Result AudioImpl::destroy() {
    if (pimpl) {
        ma_device_uninit(&pimpl->deviceCtx);
        if (pimpl->resamplerInitialized) {
            ma_resampler_uninit(&pimpl->resamplerCtx, nullptr);
        }
        pimpl.reset();
    }

    return Result::SUCCESS;
}

Result AudioImpl::reconfigure() {
    const auto& config = *candidate();
    constexpr F32 EPSILON = 1e-6f;

    if (config.deviceName != deviceName ||
        std::abs(config.inSampleRate - inSampleRate) > EPSILON ||
        std::abs(config.outSampleRate - outSampleRate) > EPSILON) {
        return Result::RECREATE;
    }

    if (std::abs(config.volume - volume) > EPSILON) {
        volume = config.volume;
        if (pimpl) {
            ma_device_set_master_volume(&pimpl->deviceCtx, volume);
        }
    }

    return Result::SUCCESS;
}

const std::string& AudioImpl::getDeviceName() const {
    return resolvedDeviceName;
}

Result AudioImpl::resample() {
    const auto& input = inputs().at("buffer").tensor;

    const F32* inputData = input.data<F32>();
    if (gatherInput) {
        U64 orderedIndex = 0;
        const U64 batchCount = batchAxis ? input.shape(*batchAxis) : 1;
        const U64 batchStride = batchAxis ? input.stride(*batchAxis) : 0;
        const U64 sampleCount = input.shape(sampleAxis);
        const U64 sampleStride = input.stride(sampleAxis);
        const U64 channelStride = channelAxis ? input.stride(*channelAxis) : 0;
        for (U64 batch = 0; batch < batchCount; ++batch) {
            for (U64 sample = 0; sample < sampleCount; ++sample) {
                for (U64 channel = 0; channel < channelCount; ++channel) {
                    orderedInput[orderedIndex++] = inputData[
                        batch * batchStride + sample * sampleStride +
                        channel * channelStride];
                }
            }
        }
    }

    const F32* currentInput = gatherInput ? orderedInput.data() : inputData;
    const U64 currentFrameCount = input.size() / channelCount;
    const U64 outputFrameCapacity = buffer.size() / channelCount;
    U64 outputFrameCount = 0;

    const auto processFrames = [&](const F32* frames,
                                   const U64 frameCount,
                                   U64& consumedFrameCount) -> Result {
        ma_uint64 frameCountIn = frameCount;
        ma_uint64 frameCountOut = outputFrameCapacity - outputFrameCount;
        F32* output = reinterpret_cast<F32*>(buffer.data()) +
                      outputFrameCount * channelCount;
        const ma_result result = ma_resampler_process_pcm_frames(
            &pimpl->resamplerCtx,
            frames,
            &frameCountIn,
            output,
            &frameCountOut
        );
        if (result != MA_SUCCESS) {
            JST_ERROR("[MODULE_AUDIO] Failed to resample audio signal.");
            return Result::ERROR;
        }

        consumedFrameCount = frameCountIn;
        outputFrameCount += frameCountOut;
        return Result::SUCCESS;
    };

    if (!pendingInput.empty()) {
        const U64 pendingFrameCount = pendingInput.size() / channelCount;
        U64 consumedPendingFrameCount = 0;
        JST_CHECK(processFrames(pendingInput.data(), pendingFrameCount,
                                consumedPendingFrameCount));

        const U64 consumedPendingSampleCount =
            consumedPendingFrameCount * channelCount;
        pendingInput.erase(pendingInput.begin(),
                           pendingInput.begin() + consumedPendingSampleCount);
        if (!pendingInput.empty()) {
            pendingInput.insert(pendingInput.end(), currentInput,
                                currentInput + input.size());
            JST_CHECK(circularBuffer.push(
                reinterpret_cast<F32*>(buffer.data()),
                outputFrameCount * channelCount));
            return Result::SUCCESS;
        }
    }

    U64 consumedCurrentFrameCount = 0;
    JST_CHECK(processFrames(currentInput, currentFrameCount,
                            consumedCurrentFrameCount));
    if (consumedCurrentFrameCount < currentFrameCount) {
        const U64 consumedCurrentSampleCount =
            consumedCurrentFrameCount * channelCount;
        pendingInput.assign(currentInput + consumedCurrentSampleCount,
                            currentInput + input.size());
    }

    JST_CHECK(circularBuffer.push(
        reinterpret_cast<F32*>(buffer.data()),
        outputFrameCount * channelCount));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
