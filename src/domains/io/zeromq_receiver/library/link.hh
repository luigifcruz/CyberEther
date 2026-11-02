#ifndef JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_LIBRARY_LINK_HH
#define JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_LIBRARY_LINK_HH

#include <any>
#include <atomic>
#include <unordered_map>
#include <string>
#include <vector>

#include <jetstream/memory/tensor.hh>
#include <jetstream/types.hh>

namespace Jetstream::ZeroMq {

class Link {
 public:
    enum class Role {
        Bind,
        Connect,
    };

    struct Config {
        int type = 0;
        Role role = Role::Bind;
        std::string address;
        U16 port = 0;
        int highWaterMark = 16;
    };

    Link();
    ~Link();

    Link(const Link&) = delete;
    Link& operator=(const Link&) = delete;

    Result open(const Config& config, std::string& error);
    void close();

    void* socket() const;
    void* monitor() const;
    const std::string& endpoint() const;

    void readEvents();
    U64 peers() const;
    bool takeHandshakeFailure();

 private:
    void* context = nullptr;
    void* handle = nullptr;
    void* events = nullptr;
    std::string address;
    U64 handshakes = 0;
    std::unordered_map<U32, U64> connections;
    std::atomic<U64> peerCount{0};
    std::atomic<bool> handshakeFailed{false};
};

class Header {
 public:
    struct Attribute {
        std::string key;
        std::any value;
    };

    struct Content {
        std::string dtype;
        std::vector<U64> shape;
        std::string stream;
        U64 sequence = 0;
        std::string signature;
        std::vector<Attribute> attributes;
    };

    static std::string Encode(const Tensor& tensor, const std::string& stream, U64 sequence);
    static bool Decode(const void* data, U64 size, Content& content);
};

}  // namespace Jetstream::ZeroMq

#endif  // JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_LIBRARY_LINK_HH
