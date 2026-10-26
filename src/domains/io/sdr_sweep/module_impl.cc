#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <new>

#include <jetstream/memory/axis.hh>
#include <jetstream/scheduler_context.hh>

namespace Jetstream::Modules {

namespace {

constexpr auto FrameLifetime = std::chrono::seconds(2);
constexpr auto MetricInterval = std::chrono::milliseconds(250);
constexpr F64 MetricSmoothing = 1.5;

std::atomic<U64>& Instances() {
    static std::atomic<U64> counter{0};
    return counter;
}

}  // namespace

void SdrSweepImpl::MetricState::reset(const Clock::time_point now, const F64 span) {
    *this = {};
    spanGHz = span / 1e9;
    occupancyUpdated = now;
    reported = now;
    rateStarted = now;
    lastCompleted = now;
}

void SdrSweepImpl::MetricState::advance(const Clock::time_point now, const bool occupied,
                                       const bool active) {
    const std::chrono::duration<F64> elapsed = now - occupancyUpdated;
    if (elapsed.count() > 0.0) {
        const F64 weight = -std::expm1(-elapsed.count() / MetricSmoothing);
        health += ((occupied ? 1.0 : 0.0) - health) * weight;
        occupancyUpdated = now;
    }
    if (now - reported >= MetricInterval) {
        const std::chrono::duration<F64> idle = now - lastCompleted;
        const F64 overdue = std::max(0.0, idle.count() - std::max(MetricSmoothing, 2.0 * cadence));
        const F64 displayed = speed * std::exp(-overdue / MetricSmoothing);
        reported = now;
        snapshot = {
            .bufferHealth = static_cast<F32>(health),
            .bufferLoss = producedFrames > 0
                ? static_cast<F64>(droppedFrames) / static_cast<F64>(producedFrames) : 0.0,
            .sweepSpeed = static_cast<F32>(displayed),
            .sweepRate = spanGHz > 0.0 ? static_cast<F32>(displayed / spanGHz) : 0.0f,
        };
    }
    if (!active) {
        snapshot.sweepSpeed = 0.0f;
        snapshot.sweepRate = 0.0f;
    }
}

void SdrSweepImpl::MetricState::publish(const Clock::time_point now, const bool occupied) {
    ++producedFrames;
    if (occupied) {
        ++droppedFrames;
    }
    lastCompleted = now;
    if (now > rateStarted) {
        const std::chrono::duration<F64> elapsedRate = now - rateStarted;
        const F64 frames = static_cast<F64>(producedFrames - sampledFrames);
        const F64 rate = frames * spanGHz / elapsedRate.count();
        speed += (rate - speed) * -std::expm1(-elapsedRate.count() / MetricSmoothing);
        cadence = elapsedRate.count() / frames;
        sampledFrames = producedFrames;
        rateStarted = now;
    }
    advance(now, occupied);
}

void SdrSweepImpl::MetricState::discard(const Clock::time_point now, const bool occupied) {
    if (occupied) {
        ++droppedFrames;
    }
    advance(now, occupied);
}

SdrSweepImpl::~SdrSweepImpl() {
    stopWorker();
}

Result SdrSweepImpl::validate() {
    const auto& config = *candidate();

    if (!std::isfinite(config.startFrequency) || !std::isfinite(config.endFrequency) ||
        config.startFrequency < 0.0 || config.startFrequency >= config.endFrequency) {
        JST_ERROR("[MODULE_SDR_SWEEP] Start frequency must be finite and below the end "
                  "frequency.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.resolution) || config.resolution <= 0.0) {
        JST_ERROR("[MODULE_SDR_SWEEP] Resolution must be finite and positive.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.sampleRate) || config.sampleRate < 0.0) {
        JST_ERROR("[MODULE_SDR_SWEEP] Capture rate must be finite and not negative.");
        return Result::ERROR;
    }
    if (config.dwell == 0 || config.dwell > Radio::SweepPlan::MaxDwell) {
        JST_ERROR("[MODULE_SDR_SWEEP] Dwell must be between 1 and {}.", Radio::SweepPlan::MaxDwell);
        return Result::ERROR;
    }
    if (!std::isfinite(config.settlingTime) || config.settlingTime < 0.0f) {
        JST_ERROR("[MODULE_SDR_SWEEP] Settling time must be finite and not negative.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.manualGain)) {
        JST_ERROR("[MODULE_SDR_SWEEP] Manual gain must be finite.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SdrSweepImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));
    return Result::SUCCESS;
}

Result SdrSweepImpl::create() {
    if (deviceString.empty()) {
        JST_ERROR("[MODULE_SDR_SWEEP] No device selected.");
        return Result::INCOMPLETE;
    }

    errored = false;
    running = false;
    completedHops.publish(0);
    sweeps.publish(0);
    expiredFrames.publish(0);
    sweepDuration.publish(0.0);
    slotFresh = false;
    sweepCounter = 0;
    instance = ++Instances();

    JST_CHECK(Radio::Sweep::Open(deviceString, radio));
    const auto& caps = radio->capabilities();
    antennas = caps.antennas;

    Radio::SweepPlan::Request request{
        .startFrequency = startFrequency,
        .endFrequency = endFrequency,
        .sampleRate = sampleRate,
        .resolution = resolution,
        .dwell = dwell,
        .settlingTime = settlingTime / 1e3,
    };
    if (const auto result = Radio::SweepPlan::Create(request, caps, plan);
        result != Result::SUCCESS) {
        radio.reset();
        return result;
    }

    const Radio::Sweep::Settings settings{
        .sampleRate = plan.sampleRate,
        .antenna = antenna,
        .automaticGain = automaticGain,
        .biasTee = biasTee,
        .settlingTime = settlingTime > 0.0f ? settlingTime / 1e3 : -1.0,
        .gain = manualGain,
    };
    if (const auto result = radio->configure(settings); result != Result::SUCCESS) {
        radio.reset();
        return result;
    }

    const F64 appliedRate = radio->applied().sampleRate;
    if (appliedRate != plan.sampleRate) {
        if (plan.mode == Radio::SweepPlan::Mode::Firmware) {
            JST_ERROR("[MODULE_SDR_SWEEP] Firmware sweeps need {:.3f} MS/s, but the radio "
                      "applied {:.6f} MS/s.", plan.sampleRate / 1e6, appliedRate / 1e6);
            radio.reset();
            return Result::ERROR;
        }
        JST_INFO("[MODULE_SDR_SWEEP] The radio applied {:.6f} MS/s instead of {:.6f} MS/s. "
                 "Replanning the sweep grid.", appliedRate / 1e6, plan.sampleRate / 1e6);
        request.sampleRate = appliedRate;
        request.mode = plan.mode;
        if (const auto result = Radio::SweepPlan::Create(request, caps, plan);
            result != Result::SUCCESS) {
            radio.reset();
            return result;
        }
    }

    try {
        JST_CHECK(buffer.create(device(), DataType::CF32, {1, plan.outputBins}));
        JST_CHECK(SetSignalAxes(buffer, {
            .sample = Index{1},
            .batch = Index{0},
        }));
        staging.assign(plan.outputBins, CF32{});
        slot.assign(plan.outputBins, CF32{});
    } catch (const std::bad_alloc&) {
        JST_ERROR("[MODULE_SDR_SWEEP] Failed to allocate the sweep buffers.");
        radio.reset();
        return Result::ERROR;
    }
    JST_CHECK(assembler.configure(plan));
    totalHops.publish(plan.hops.size());

    if (automaticGain && radio->applied().automaticGain) {
        JST_WARN("[MODULE_SDR_SWEEP] Automatic gain is on. Levels of different hops are not "
                 "directly comparable.");
    }
    if (const auto result = radio->start(plan.radioRequest()); result != Result::SUCCESS) {
        radio.reset();
        return result;
    }
    gainRange = radio->applied().gainRange;
    if (gainRange) {
        manualGain = static_cast<F32>(radio->applied().gain);
    }

    JST_CHECK(buffer.setAttribute("frequency", static_cast<F32>(plan.centerFrequency())));
    JST_CHECK(buffer.setAttribute("sampleRate", static_cast<F32>(plan.span())));
    JST_CHECK(buffer.setAttribute("sweepSynthetic", true));
    JST_CHECK(buffer.setAttribute("sweepMethod", Radio::SweepPlan::ModeName(plan.mode)));
    JST_CHECK(buffer.setAttribute("sweepStartFrequency", plan.startFrequency));
    JST_CHECK(buffer.setAttribute("sweepEndFrequency", plan.endFrequency()));
    JST_CHECK(buffer.setAttribute("sweepBinWidth", plan.binWidth));
    JST_CHECK(buffer.setAttribute("sweepRequestedResolution", resolution));
    JST_CHECK(buffer.setAttribute("sweepOutputBins", plan.outputBins));
    JST_CHECK(buffer.setAttribute("sweepCaptureRate", plan.sampleRate));
    JST_CHECK(buffer.setAttribute("sweepGeneration", instance));
    JST_CHECK(buffer.setAttribute("sweepId", U64{0}));
    JST_CHECK(buffer.setAttribute("sweepHostDuration", 0.0));

    outputs()["signal"].produced(name(), "signal", buffer);

    JST_INFO("[MODULE_SDR_SWEEP] Sweeping {:.3f} to {:.3f} MHz in {} hops of {} samples at "
             "{:.3f} MS/s with {} bins of {:.1f} Hz ({}).", plan.startFrequency / 1e6,
             plan.endFrequency() / 1e6, plan.hops.size(), plan.segmentLength,
             plan.sampleRate / 1e6, plan.outputBins, plan.binWidth,
             Radio::SweepPlan::ModeName(plan.mode));

    if (const auto result = startWorker(); result != Result::SUCCESS) {
        radio.reset();
        return result;
    }
    return Result::SUCCESS;
}

Result SdrSweepImpl::destroy() {
    stopWorker();
    radio.reset();
    return Result::SUCCESS;
}

Result SdrSweepImpl::reconfigure() {
    const auto& config = *candidate();
    if (!radio || errored || config.deviceString != deviceString ||
        config.antenna != antenna || config.biasTee != biasTee ||
        config.startFrequency != startFrequency || config.endFrequency != endFrequency ||
        config.sampleRate != sampleRate || config.resolution != resolution) {
        return Result::RECREATE;
    }

    return access.change([&] {
        Radio::SweepPlan next;
        const Radio::SweepPlan::Request request{
            .startFrequency = config.startFrequency,
            .endFrequency = config.endFrequency,
            .sampleRate = plan.sampleRate,
            .resolution = config.resolution,
            .dwell = config.dwell,
            .settlingTime = config.settlingTime / 1e3,
        };
        if (Radio::SweepPlan::Create(request, radio->capabilities(), next) != Result::SUCCESS ||
            next.mode != plan.mode || next.sampleRate != plan.sampleRate ||
            next.segmentLength != plan.segmentLength || next.binWidth != plan.binWidth ||
            next.startFrequency != plan.startFrequency ||
            next.outputBins != plan.outputBins || next.hops.size() != plan.hops.size()) {
            return Result::RECREATE;
        }

        auto settings = radio->settings();
        settings.automaticGain = config.automaticGain;
        settings.gain = config.manualGain;
        settings.settlingTime = config.settlingTime > 0.0f ? config.settlingTime / 1e3 : -1.0;
        JST_CHECK(applyPlan(next, settings));

        automaticGain = config.automaticGain;
        manualGain = gainRange ? static_cast<F32>(radio->applied().gain)
                               : config.manualGain;
        dwell = config.dwell;
        settlingTime = config.settlingTime;
        return Result::SUCCESS;
    });
}

Result SdrSweepImpl::applyPlan(const Radio::SweepPlan& next,
                               const Radio::Sweep::Settings& settings) {
    const auto previous = radio->settings();
    if (const auto result = radio->update(settings); result != Result::SUCCESS) {
        if (result == Result::FATAL) {
            errored = true;
            return Result::ERROR;
        }
        return result;
    }
    const auto request = next.radioRequest();
    const auto result = [&] {
        JST_CHECK(assembler.configure(next));
        if (request != radio->request()) {
            JST_CHECK(radio->start(request));
        }
        return Result::SUCCESS;
    }();
    if (result != Result::SUCCESS) {
        const auto current = plan.radioRequest();
        const auto restoredSweep = radio->request() == current ? Result::SUCCESS
                                                               : radio->start(current);
        const auto restoredSettings = radio->update(previous);
        const auto restoredAssembler = assembler.configure(plan);
        if (restoredSettings != Result::SUCCESS || restoredAssembler != Result::SUCCESS ||
            restoredSweep != Result::SUCCESS) {
            errored = true;
            JST_ERROR("[MODULE_SDR_SWEEP] Failed to restore the previous sweep settings.");
        }
        return result;
    }
    plan = next;
    completedHops.publish(0);
    {
        std::lock_guard lock(slotMutex);
        metrics.discard(Clock::now(), slotFresh);
        slotFresh = false;
    }
    return Result::SUCCESS;
}

Result SdrSweepImpl::startWorker() {
    running = true;
    {
        std::lock_guard lock(slotMutex);
        metrics.reset(Clock::now(), plan.span());
    }
    try {
        worker = std::thread([this] {
            Result result = Result::ERROR;
            try {
                result = workerLoop();
            } catch (...) {
                result = Result::ERROR;
            }
            if (result != Result::SUCCESS) {
                errored = true;
            }
            slotReady.notify_all();
            Scheduler::Context::NotifyPendingCompute();
        });
    } catch (const std::exception& e) {
        running = false;
        JST_ERROR("[MODULE_SDR_SWEEP] Failed to start the sweep thread: {}", e.what());
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

void SdrSweepImpl::stopWorker() {
    {
        std::lock_guard lock(slotMutex);
        running = false;
    }
    if (worker.joinable()) {
        worker.join();
    }
    {
        std::lock_guard lock(slotMutex);
        slotFresh = false;
        metrics.reset(Clock::now());
    }
}

Result SdrSweepImpl::workerLoop() {
    while (running) {
        const auto lock = access.lock();
        if (!running) {
            break;
        }
        if (errored) {
            return Result::ERROR;
        }
        const auto result = radio->run(
            [this] { return running.load() && !access.interrupted(); },
            [this](const auto& capture) { return handleCapture(capture); });
        if (errored) {
            return Result::ERROR;
        }
        JST_CHECK(result);
    }
    return Result::SUCCESS;
}

bool SdrSweepImpl::handleCapture(const Radio::Sweep::Capture& capture) {
    const auto now = Clock::now();
    if (capture.hop == 0) {
        sweepStarted = now;
    }

    const auto result = assembler.add(capture);
    if (result == Result::INCOMPLETE) {
        completedHops.publish(0);
        return running;
    }
    if (result != Result::SUCCESS) {
        errored = true;
        return false;
    }
    completedHops.publish(assembler.completedHops());

    if (!assembler.complete()) {
        return running;
    }
    if (assembler.synthesize(staging) != Result::SUCCESS) {
        errored = true;
        return false;
    }

    return publishFrame(Clock::now());
}

bool SdrSweepImpl::publishFrame(const Clock::time_point now) {
    {
        std::lock_guard lock(slotMutex);
        if (!running) {
            return false;
        }
        expireFrame(now);
        metrics.publish(now, slotFresh);
        std::swap(slot, staging);
        slotFrame = {.sweep = ++sweepCounter, .started = sweepStarted, .completed = now};
        slotFresh = true;
    }
    slotReady.notify_all();
    Scheduler::Context::NotifyPendingCompute();
    sweeps.publish(sweepCounter);
    sweepDuration.publish(std::chrono::duration<F64>(now - sweepStarted).count());
    return running;
}

bool SdrSweepImpl::frameExpired(const Frame& frame, const Clock::time_point now) const {
    const auto lifetime = std::max<std::chrono::duration<F64>>(
        FrameLifetime, std::chrono::duration<F64>(2.0 * sweepDuration.get()));
    return now - frame.completed > lifetime;
}

void SdrSweepImpl::expireFrame(const Clock::time_point now) {
    if (slotFresh && frameExpired(slotFrame, now)) {
        const auto lifetime = std::max<std::chrono::duration<F64>>(
            FrameLifetime, std::chrono::duration<F64>(2.0 * sweepDuration.get()));
        const auto expired = slotFrame.completed + std::chrono::duration_cast<Clock::duration>(lifetime);
        metrics.discard(expired, true);
        slotFresh = false;
        expiredFrames.publish(expiredFrames.get() + 1);
    }
}

Result SdrSweepImpl::emitFrame(const Clock::time_point now) {
    Frame frame;
    {
        std::lock_guard lock(slotMutex);
        expireFrame(now);
        if (!slotFresh) {
            return Result::YIELD;
        }
        std::copy(slot.begin(), slot.end(), buffer.data<CF32>());
        frame = slotFrame;
        metrics.advance(now, true);
        slotFresh = false;
    }
    JST_CHECK(buffer.setAttribute("sweepId", frame.sweep));
    JST_CHECK(buffer.setAttribute("sweepHostDuration",
                                  std::chrono::duration<F64>(frame.completed - frame.started).count()));
    return Result::SUCCESS;
}

SdrSweepImpl::Progress SdrSweepImpl::getProgress() const {
    return {
        .completedHops = completedHops.get(),
        .totalHops = totalHops.get(),
        .sweeps = sweeps.get(),
        .failedHops = radio ? radio->statistics().failedHops : 0,
        .expiredFrames = expiredFrames.get(),
    };
}

SdrSweepImpl::MetricSnapshot SdrSweepImpl::getMetrics(const Clock::time_point now) {
    std::lock_guard lock(slotMutex);
    expireFrame(now);
    metrics.advance(now, slotFresh, running && !errored);
    return metrics.snapshot;
}

std::vector<std::string> SdrSweepImpl::listAntennas() const {
    return antennas;
}

std::optional<Radio::Device::Range> SdrSweepImpl::getGainRange() const {
    return gainRange;
}

const Radio::SweepPlan& SdrSweepImpl::sweepPlan() const {
    return plan;
}

}  // namespace Jetstream::Modules
