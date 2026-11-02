#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_RECEIVER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_RECEIVER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct WebsocketReceiver : public Module::Config {
    std::string url = "ws://127.0.0.1:8765";
    std::string dataType = "CF32";
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_MODULE_TYPE(websocket_receiver);
    JST_MODULE_PARAMS(url, dataType, numberOfBatches, numberOfTimeSamples,
                      bufferMultiplier);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_RECEIVER_MODULE_HH
