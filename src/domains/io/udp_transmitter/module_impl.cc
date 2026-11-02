#include "module_impl.hh"

#include <limits>

namespace Jetstream::Modules {

namespace {

constexpr U64 kMaxPacketBytes = 65507;
constexpr U64 kSocketBufferBytes = 8 * JST_MB;
constexpr U64 kSendTimeoutMilliseconds = 100;

}  // namespace

Result UdpTransmitterImpl::validate() {
    const auto& config = *candidate();

    if (config.port == 0 || config.port > std::numeric_limits<U16>::max()) {
        JST_ERROR("[MODULE_UDP_TRANSMITTER] Port {} is outside the range 1 to 65535.",
                  config.port);
        return Result::ERROR;
    }

    if (config.packetSize == 0 || config.packetSize > kMaxPacketBytes) {
        JST_ERROR("[MODULE_UDP_TRANSMITTER] Packet size {} is outside the range 1 to {} bytes.",
                  config.packetSize, kMaxPacketBytes);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result UdpTransmitterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));

    return Result::SUCCESS;
}

Result UdpTransmitterImpl::create() {
    if (address.empty()) {
        JST_WARN("[MODULE_UDP_TRANSMITTER] Destination address is empty.");
        return Result::INCOMPLETE;
    }

    throughputMBs.publish(0.0f);
    droppedPackets.publish(0);
    warnedDrop = false;
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    const auto& input = inputs().at("buffer").tensor;
    const U64 elementSize = input.elementSize();
    if (elementSize == 0 || packetSize < elementSize) {
        JST_ERROR("[MODULE_UDP_TRANSMITTER] Packet size {} is smaller than one {} sample.",
                  packetSize, input.dtype());
        return Result::ERROR;
    }
    payloadBytes = packetSize - packetSize % elementSize;

    std::string error;
    const Platform::UdpSocket::Config socketConfig = {
        .address = address,
        .port = static_cast<U16>(port),
        .interfaceName = networkInterface == "auto" ? std::string() : networkInterface,
        .bufferBytes = kSocketBufferBytes,
        .timeoutMilliseconds = kSendTimeoutMilliseconds,
    };
    if (socket.connect(socketConfig, error) != Result::SUCCESS) {
        JST_ERROR("[MODULE_UDP_TRANSMITTER] Failed to open a socket to '{}' port {} ({}).",
                  address, port, error);
        return Result::ERROR;
    }

    JST_INFO("[MODULE_UDP_TRANSMITTER] Sending to '{}' port {}.", address, port);

    return Result::SUCCESS;
}

Result UdpTransmitterImpl::destroy() {
    socket.close();

    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    return Result::SUCCESS;
}

F32 UdpTransmitterImpl::getThroughput() const {
    return throughputMBs.get();
}

U64 UdpTransmitterImpl::getDroppedPackets() const {
    return droppedPackets.get();
}

void UdpTransmitterImpl::updateThroughput(const U64 deltaBytes) {
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
