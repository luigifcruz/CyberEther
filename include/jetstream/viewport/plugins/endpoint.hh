#ifndef JETSTREAM_VIEWPORT_PLUGINS_ENDPOINT_HH
#define JETSTREAM_VIEWPORT_PLUGINS_ENDPOINT_HH

#include "jetstream/viewport/adapters/generic.hh"

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
#include <gst/gst.h>
#endif

#include <thread>
#include <mutex>
#include <condition_variable>

namespace Jetstream::Viewport {

class Endpoint {
 public:
    Result create(const Viewport::Config& config);
    Result destroy();

    Result newFrameHost(const uint8_t* data);

 private:
   enum class Type {
      File,
      Pipe,
      Socket,
      Unknown
   };

   Config config;
   Endpoint::Type type;

   Result createFileEndpoint();
   Result createPipeEndpoint();
   Result createSocketEndpoint();

   Result destroyFileEndpoint();
   Result destroyPipeEndpoint();
   Result destroySocketEndpoint();

   static Endpoint::Type DetermineEndpointType(const std::string& endpoint);

   // Pipe endpoint.
   int pipeFileDescriptor;
   bool pipeCreated = false;

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
   GstElement* pipeline;
   GstElement* source;

   std::mutex bufferMutex;
   std::condition_variable bufferCond;
   bool bufferProcessed = false;

   static void OnBufferReleaseCallback(gpointer user_data);

   // Broker endpoint.
   std::string brokerAddress;
   int brokerPort;
   int brokerEndpointFileDescriptor;
   int brokerClientFileDescriptor;
   bool brokerEndpointRunning;
   std::thread brokerEndpointThread;

   // Socket endpoint.
   std::string socketAddress;
   int socketPort;
   bool socketConnected = false;
   bool socketStreaming = false;

   // File endpoint. 
   std::string fileExtension;
   
   Result createGstreamerEndpoint();
   Result destroyGstreamerEndpoint();
#endif
};

}  // namespace Jetstream::Viewport

#endif