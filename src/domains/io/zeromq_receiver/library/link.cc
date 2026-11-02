#include "link.hh"

#include <cstring>
#include <typeinfo>

#include <nlohmann/json.hpp>
#include <zmq.h>

#include <jetstream/memory/types.hh>

namespace Jetstream::ZeroMq {

namespace {

constexpr const char* kMonitorEndpoint = "inproc://monitor";
constexpr int kReconnectMilliseconds = 500;
constexpr int kHeartbeatMilliseconds = 1000;
constexpr int kHeartbeatTimeoutMilliseconds = 5000;
constexpr int kHandshakeMilliseconds = 5000;
constexpr I64 kMaxMessageBytes = 256LL * 1024 * 1024;
constexpr int kHeaderVersion = 1;

template<typename T>
bool SetOption(void* socket, const int option, const T& value) {
    return zmq_setsockopt(socket, option, &value, sizeof(value)) == 0;
}

std::string LastError() {
    return zmq_strerror(zmq_errno());
}

template<typename T>
bool EncodeScalar(const std::any& value, const char* type, nlohmann::json& entry) {
    if (value.type() != typeid(T)) {
        return false;
    }
    entry = {{"type", type}, {"value", std::any_cast<T>(value)}};
    return true;
}

template<typename T>
bool EncodeComplex(const std::any& value, const char* type, nlohmann::json& entry) {
    if (value.type() != typeid(T)) {
        return false;
    }
    const auto& complex = std::any_cast<const T&>(value);
    entry = {{"type", type}, {"value", {complex.real(), complex.imag()}}};
    return true;
}

template<typename T>
bool EncodeVector(const std::any& value, const char* type, nlohmann::json& entry) {
    if (value.type() != typeid(std::vector<T>)) {
        return false;
    }
    entry = {{"type", type}, {"value", std::any_cast<const std::vector<T>&>(value)}};
    return true;
}

bool EncodeAttribute(const std::any& value, nlohmann::json& entry) {
    return EncodeScalar<bool>(value, "bool", entry) ||
           EncodeScalar<std::string>(value, "string", entry) ||
           EncodeScalar<I8>(value, "I8", entry) ||
           EncodeScalar<I16>(value, "I16", entry) ||
           EncodeScalar<I32>(value, "I32", entry) ||
           EncodeScalar<I64>(value, "I64", entry) ||
           EncodeScalar<U8>(value, "U8", entry) ||
           EncodeScalar<U16>(value, "U16", entry) ||
           EncodeScalar<U32>(value, "U32", entry) ||
           EncodeScalar<U64>(value, "U64", entry) ||
           EncodeScalar<F32>(value, "F32", entry) ||
           EncodeScalar<F64>(value, "F64", entry) ||
           EncodeComplex<CF32>(value, "CF32", entry) ||
           EncodeComplex<CF64>(value, "CF64", entry) ||
           EncodeVector<I64>(value, "vector<I64>", entry) ||
           EncodeVector<U64>(value, "vector<U64>", entry) ||
           EncodeVector<F32>(value, "vector<F32>", entry) ||
           EncodeVector<F64>(value, "vector<F64>", entry);
}

bool DecodeAttribute(const nlohmann::json& entry, std::any& value) {
    if (!entry.is_object() || !entry.contains("type") || !entry.contains("value") ||
        !entry["type"].is_string()) {
        return false;
    }

    const auto type = entry["type"].get<std::string>();
    const auto& data = entry["value"];

    const auto complex = [&]<typename T>(T) {
        if (!data.is_array() || data.size() != 2 || !data[0].is_number() || !data[1].is_number()) {
            return false;
        }
        value = T(data[0].get<typename T::value_type>(), data[1].get<typename T::value_type>());
        return true;
    };

    try {
        if (type == "bool" && data.is_boolean()) {
            value = data.get<bool>();
        } else if (type == "string" && data.is_string()) {
            value = data.get<std::string>();
        } else if (type == "I8" && data.is_number_integer()) {
            value = data.get<I8>();
        } else if (type == "I16" && data.is_number_integer()) {
            value = data.get<I16>();
        } else if (type == "I32" && data.is_number_integer()) {
            value = data.get<I32>();
        } else if (type == "I64" && data.is_number_integer()) {
            value = data.get<I64>();
        } else if (type == "U8" && data.is_number_unsigned()) {
            value = data.get<U8>();
        } else if (type == "U16" && data.is_number_unsigned()) {
            value = data.get<U16>();
        } else if (type == "U32" && data.is_number_unsigned()) {
            value = data.get<U32>();
        } else if (type == "U64" && data.is_number_unsigned()) {
            value = data.get<U64>();
        } else if (type == "F32" && data.is_number()) {
            value = data.get<F32>();
        } else if (type == "F64" && data.is_number()) {
            value = data.get<F64>();
        } else if (type == "CF32") {
            return complex(CF32{});
        } else if (type == "CF64") {
            return complex(CF64{});
        } else if (type == "vector<I64>" && data.is_array()) {
            value = data.get<std::vector<I64>>();
        } else if (type == "vector<U64>" && data.is_array()) {
            value = data.get<std::vector<U64>>();
        } else if (type == "vector<F32>" && data.is_array()) {
            value = data.get<std::vector<F32>>();
        } else if (type == "vector<F64>" && data.is_array()) {
            value = data.get<std::vector<F64>>();
        } else {
            return false;
        }
    } catch (const nlohmann::json::exception&) {
        return false;
    }
    return true;
}

}  // namespace

Link::Link() = default;

Link::~Link() {
    close();
}

Result Link::open(const Config& config, std::string& error) {
    close();

    address = "tcp://";
    address += config.address.find(':') != std::string::npos ? "[" + config.address + "]" : config.address;
    address += ":" + std::to_string(config.port);

    context = zmq_ctx_new();
    if (!context) {
        error = LastError();
        return Result::ERROR;
    }

    handle = zmq_socket(context, config.type);
    if (!handle) {
        error = LastError();
        close();
        return Result::ERROR;
    }

    const int zero = 0;
    const int one = 1;
    if (!SetOption(handle, ZMQ_LINGER, zero) ||
        !SetOption(handle, ZMQ_SNDHWM, config.highWaterMark) ||
        !SetOption(handle, ZMQ_RCVHWM, config.highWaterMark) ||
        !SetOption(handle, ZMQ_IPV6, one) ||
        !SetOption(handle, ZMQ_RECONNECT_IVL, kReconnectMilliseconds) ||
        !SetOption(handle, ZMQ_RECONNECT_IVL_MAX, kReconnectMilliseconds) ||
        !SetOption(handle, ZMQ_HEARTBEAT_IVL, kHeartbeatMilliseconds) ||
        !SetOption(handle, ZMQ_HEARTBEAT_TIMEOUT, kHeartbeatTimeoutMilliseconds) ||
        !SetOption(handle, ZMQ_HANDSHAKE_IVL, kHandshakeMilliseconds) ||
        !SetOption(handle, ZMQ_TCP_KEEPALIVE, one) ||
        !SetOption(handle, ZMQ_MAXMSGSIZE, kMaxMessageBytes)) {
        error = LastError();
        close();
        return Result::ERROR;
    }

    if (config.type == ZMQ_PUSH && !SetOption(handle, ZMQ_IMMEDIATE, one)) {
        error = LastError();
        close();
        return Result::ERROR;
    }

    if (config.type == ZMQ_SUB && zmq_setsockopt(handle, ZMQ_SUBSCRIBE, "", 0) != 0) {
        error = LastError();
        close();
        return Result::ERROR;
    }

    const int watched = ZMQ_EVENT_CONNECTED | ZMQ_EVENT_ACCEPTED | ZMQ_EVENT_DISCONNECTED |
                        ZMQ_EVENT_HANDSHAKE_SUCCEEDED | ZMQ_EVENT_HANDSHAKE_FAILED_PROTOCOL;
    if (zmq_socket_monitor(handle, kMonitorEndpoint, watched) != 0) {
        error = LastError();
        close();
        return Result::ERROR;
    }

    events = zmq_socket(context, ZMQ_PAIR);
    if (!events || !SetOption(events, ZMQ_LINGER, zero) || zmq_connect(events, kMonitorEndpoint) != 0) {
        error = LastError();
        close();
        return Result::ERROR;
    }

    const int attached = config.role == Role::Bind ? zmq_bind(handle, address.c_str())
                                                   : zmq_connect(handle, address.c_str());
    if (attached != 0) {
        error = LastError();
        close();
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

void Link::close() {
    if (handle) {
        zmq_socket_monitor(handle, nullptr, 0);
    }
    if (events) {
        zmq_close(events);
        events = nullptr;
    }
    if (handle) {
        zmq_close(handle);
        handle = nullptr;
    }
    if (context) {
        zmq_ctx_term(context);
        context = nullptr;
    }
    handshakes = 0;
    connections.clear();
    peerCount = 0;
    handshakeFailed = false;
}

void* Link::socket() const {
    return handle;
}

void* Link::monitor() const {
    return events;
}

const std::string& Link::endpoint() const {
    return address;
}

void Link::readEvents() {
    if (!events) {
        return;
    }

    while (true) {
        zmq_msg_t frame;
        zmq_msg_init(&frame);
        if (zmq_msg_recv(&frame, events, ZMQ_DONTWAIT) < 0) {
            zmq_msg_close(&frame);
            return;
        }

        U16 event = 0;
        U32 value = 0;
        if (zmq_msg_size(&frame) >= 6) {
            const auto* bytes = static_cast<const U8*>(zmq_msg_data(&frame));
            std::memcpy(&event, bytes, sizeof(event));
            std::memcpy(&value, bytes + sizeof(event), sizeof(value));
        }
        bool more = zmq_msg_more(&frame) != 0;
        zmq_msg_close(&frame);

        while (more) {
            zmq_msg_init(&frame);
            more = zmq_msg_recv(&frame, events, ZMQ_DONTWAIT) >= 0 && zmq_msg_more(&frame) != 0;
            zmq_msg_close(&frame);
        }

        if (event == ZMQ_EVENT_CONNECTED || event == ZMQ_EVENT_ACCEPTED) {
            connections[value] = handshakes;
        } else if (event == ZMQ_EVENT_HANDSHAKE_SUCCEEDED) {
            handshakes += 1;
        } else if (event == ZMQ_EVENT_DISCONNECTED) {
            const auto connection = connections.find(value);
            if (connection != connections.end()) {
                if (connection->second == handshakes) {
                    handshakeFailed = true;
                }
                connections.erase(connection);
            }
        } else if (event == ZMQ_EVENT_HANDSHAKE_FAILED_PROTOCOL) {
            handshakeFailed = true;
        }
        peerCount = connections.size();
    }
}

U64 Link::peers() const {
    return peerCount;
}

bool Link::takeHandshakeFailure() {
    return handshakeFailed.exchange(false);
}

std::string Header::Encode(const Tensor& tensor, const std::string& stream, const U64 sequence) {
    nlohmann::json attributes = nlohmann::json::object();
    for (const auto& key : tensor.attributeKeys()) {
        nlohmann::json entry;
        if (EncodeAttribute(tensor.attribute(key), entry)) {
            attributes[key] = std::move(entry);
        }
    }

    nlohmann::json shape = nlohmann::json::array();
    for (const auto dimension : tensor.shape()) {
        shape.push_back(dimension);
    }

    const nlohmann::json header = {
        {"cyberether", kHeaderVersion},
        {"stream", stream},
        {"sequence", sequence},
        {"dtype", std::string(DataTypeToName(tensor.dtype()))},
        {"shape", std::move(shape)},
        {"attributes", std::move(attributes)},
    };
    return header.dump();
}

bool Header::Decode(const void* data, const U64 size, Content& content) {
    const auto* text = static_cast<const char*>(data);
    if (size < 2 || text[0] != '{') {
        return false;
    }

    const auto json = nlohmann::json::parse(text, text + size, nullptr, false);
    if (json.is_discarded() || !json.is_object() || !json.contains("cyberether") ||
        !json["cyberether"].is_number_integer() || json["cyberether"].get<I64>() != kHeaderVersion) {
        return false;
    }

    content = {};
    if (json.contains("dtype") && json["dtype"].is_string()) {
        content.dtype = json["dtype"].get<std::string>();
    }
    if (json.contains("stream") && json["stream"].is_string()) {
        content.stream = json["stream"].get<std::string>();
    }
    if (json.contains("sequence") && json["sequence"].is_number_unsigned()) {
        content.sequence = json["sequence"].get<U64>();
    }
    if (json.contains("shape") && json["shape"].is_array()) {
        for (const auto& dimension : json["shape"]) {
            if (dimension.is_number_unsigned()) {
                content.shape.push_back(dimension.get<U64>());
            }
        }
    }
    if (json.contains("attributes") && json["attributes"].is_object()) {
        content.signature = json["attributes"].dump();
        for (const auto& [key, entry] : json["attributes"].items()) {
            std::any value;
            if (DecodeAttribute(entry, value)) {
                content.attributes.push_back({key, std::move(value)});
            }
        }
    }
    return true;
}

}  // namespace Jetstream::ZeroMq
