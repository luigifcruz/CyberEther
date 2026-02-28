#ifndef JETSTREAM_DOMAINS_IO_AUDIO_MODULE_HH
#define JETSTREAM_DOMAINS_IO_AUDIO_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Audio : public Module::Config {
    std::string deviceName = "Default";
    F32 inSampleRate = 48e3;
    F32 outSampleRate = 48e3;
    F32 volume = 1.0f;

    JST_MODULE_TYPE(audio);
    JST_MODULE_PARAMS(deviceName, inSampleRate, outSampleRate, volume);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_AUDIO_MODULE_HH
