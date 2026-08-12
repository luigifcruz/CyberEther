#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_TEST_SERVER_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_TEST_SERVER_HH

#include <jetstream/config.hh>

#ifndef JST_OS_BROWSER

#include <string>
#include <thread>

#include <httplib.h>

namespace Jetstream::Tests {

class WebsocketTestServer {
 public:
    WebsocketTestServer() {
        server.WebSocket("/", [](const httplib::Request&,
                                 httplib::ws::WebSocket&) {});
        port = server.bind_to_any_port("127.0.0.1");
        if (port > 0) {
            thread = std::thread([this]() { server.listen_after_bind(); });
            server.wait_until_ready();
        }
    }

    ~WebsocketTestServer() {
        server.stop();
        if (thread.joinable()) {
            thread.join();
        }
    }

    bool valid() const {
        return port > 0;
    }

    std::string url() const {
        return "ws://127.0.0.1:" + std::to_string(port);
    }

 private:
    httplib::Server server;
    std::thread thread;
    int port = -1;
};

}  // namespace Jetstream::Tests

#endif  // JST_OS_BROWSER

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_TEST_SERVER_HH
