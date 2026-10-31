#include "module_impl.hh"

#include <limits>
#include <random>

#include <zmq.h>

namespace Jetstream::Modules {

namespace {

constexpr long kPollMilliseconds = 200;

}  // namespace

Result ZeromqTransmitterImpl::validate() {
    const auto& config = *candidate();

    if (config.pattern != "publish" && config.pattern != "push") {
        JST_ERROR("[MODULE_ZEROMQ_TRANSMITTER] Invalid pattern '{}'.", config.pattern);
        return Result::ERROR;
    }

    if (config.mode != "bind" && config.mode != "connect") {
        JST_ERROR("[MODULE_ZEROMQ_TRANSMITTER] Invalid mode '{}'.", config.mode);
        return Result::ERROR;
    }

    if (config.port == 0 || config.port > std::numeric_limits<U16>::max()) {
        JST_ERROR("[MODULE_ZEROMQ_TRANSMITTER] Port {} is outside the range 1 to 65535.",
                  config.port);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ZeromqTransmitterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));

    return Result::SUCCESS;
}

Result ZeromqTransmitterImpl::create() {
    if (address.empty()) {
        JST_WARN("[MODULE_ZEROMQ_TRANSMITTER] Address is empty.");
        return Result::INCOMPLETE;
    }

    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();
    droppedBuffers = 0;
    warnedDrop = false;
    sequence = 0;

    std::random_device entropy;
    stream = jst::fmt::format("{:08x}{:08x}", entropy(), entropy());

    const bool bind = mode == "bind";
    std::string error;
    const ZeroMq::Link::Config config = {
        .type = pattern == "push" ? ZMQ_PUSH : ZMQ_PUB,
        .role = bind ? ZeroMq::Link::Role::Bind : ZeroMq::Link::Role::Connect,
        .address = address,
        .port = static_cast<U16>(port),
    };
    if (link.open(config, error) != Result::SUCCESS) {
        JST_ERROR("[MODULE_ZEROMQ_TRANSMITTER] Failed to {} '{}' port {} ({}).",
                  bind ? "listen on" : "connect to", address, port, error);
        return Result::ERROR;
    }

    if (bind) {
        JST_INFO("[MODULE_ZEROMQ_TRANSMITTER] Listening on '{}' port {}.", address, port);
    } else {
        JST_INFO("[MODULE_ZEROMQ_TRANSMITTER] Connecting to '{}' port {}.", address, port);
    }

    monitoring = true;
    monitorThread.start([this]() { monitorLoop(); });

    return Result::SUCCESS;
}

Result ZeromqTransmitterImpl::destroy() {
    monitoring = false;
    if (monitorThread.joinable()) {
        monitorThread.join();
    }
    link.close();

    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    return Result::SUCCESS;
}

void ZeromqTransmitterImpl::send(const Tensor& input) {
    void* socket = link.socket();
    const U64 size = input.sizeBytes();
    const U64 current = sequence++;

    const auto drop = [&]() {
        droppedBuffers += 1;
        if (!warnedDrop) {
            JST_WARN("[MODULE_ZEROMQ_TRANSMITTER] No receiver is ready at '{}' port {}. "
                     "Dropping buffers.", address, port);
            warnedDrop = true;
        }
    };

    if (sendAttributes) {
        const auto header = ZeroMq::Header::Encode(input, stream, current);
        if (zmq_send(socket, header.data(), header.size(), ZMQ_SNDMORE | ZMQ_DONTWAIT) < 0) {
            drop();
            return;
        }
    }

    if (zmq_send(socket, input.data(), size, ZMQ_DONTWAIT) < 0) {
        drop();
        return;
    }

    warnedDrop = false;
    updateThroughput(size);
}

void ZeromqTransmitterImpl::monitorLoop() {
    bool warned = false;
    zmq_pollitem_t items[] = {{link.monitor(), 0, ZMQ_POLLIN, 0}};

    while (monitoring) {
        if (zmq_poll(items, 1, kPollMilliseconds) <= 0) {
            continue;
        }
        link.readEvents();
        if (link.takeHandshakeFailure() && !warned) {
            JST_WARN("[MODULE_ZEROMQ_TRANSMITTER] A peer at '{}' port {} disconnected during the "
                     "handshake. Check that Pattern matches on both ends.", address, port);
            warned = true;
        }
    }
}

F32 ZeromqTransmitterImpl::getThroughput() const {
    return throughputMBs.get();
}

U64 ZeromqTransmitterImpl::getPeers() const {
    return link.peers();
}

U64 ZeromqTransmitterImpl::getDroppedBuffers() const {
    return droppedBuffers;
}

void ZeromqTransmitterImpl::updateThroughput(const U64 deltaBytes) {
    constexpr double kMeasurementPeriodSeconds = 0.10;
    constexpr double kEmaAlpha = 0.3;

    bytesSinceLastMeasurement += deltaBytes;

    const auto now = std::chrono::steady_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(now - lastMeasurementTime).count();
    if (elapsedSeconds < kMeasurementPeriodSeconds) {
        return;
    }

    const double instantThroughput = static_cast<double>(bytesSinceLastMeasurement) /
                                     1e6 /
                                     elapsedSeconds;
    const double smoothedThroughput = kEmaAlpha * instantThroughput +
                                      (1.0 - kEmaAlpha) * throughputMBs.get();
    throughputMBs.publish(static_cast<F32>(smoothedThroughput));

    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = now;
}

}  // namespace Jetstream::Modules
