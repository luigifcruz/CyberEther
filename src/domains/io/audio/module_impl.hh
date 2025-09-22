#ifndef JETSTREAM_DOMAINS_IO_AUDIO_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_AUDIO_MODULE_IMPL_HH

#include <vector>
#include <string>
#include <memory>

#include <jetstream/domains/io/audio/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/circular_buffer.hh>

namespace Jetstream::Modules {

struct AudioImpl : public Module::Impl, public DynamicConfig<Audio> {
 public:
    AudioImpl();
    ~AudioImpl();

    using DeviceList = std::vector<std::string>;

    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    static DeviceList ListAvailableDevices();

    const std::string& getDeviceName() const;
    Result resample();

 protected:
    Tensor buffer;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    std::string resolvedDeviceName;

    Tools::CircularBuffer<F32> circularBuffer;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_AUDIO_MODULE_IMPL_HH
