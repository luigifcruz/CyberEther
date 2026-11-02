#ifndef JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH

#include <atomic>
#include <thread>

#include <jetstream/domains/io/soapy/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

#include "domains/io/sdr/tools/stream_access.hh"
#include "domains/io/sdr/tools/stream_buffer.hh"
#include "library/soapysdr.hh"

namespace Jetstream::Modules {

struct JETSTREAM_API SoapyImpl : public Module::Impl, public DynamicConfig<Soapy> {
 public:
    ~SoapyImpl() override;

    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    F32 getBufferFill() const;
    F64 getBufferLoss() const;
    U64 getDeviceOverflows() const;
    std::pair<F32, F32> getThroughput() const;
    SoapyDeviceInfo getDeviceInfo() const;

 protected:
    Tensor buffer;
    DataType validatedDataType = DataType::CF32;

    U64 validatedOutputSizeBytes = 0;
    U64 validatedInternalElements = 0;
    U64 validatedInternalSizeBytes = 0;

    std::thread producer;
    std::atomic<bool> errored{false};
    std::atomic<bool> streaming{false};

    Radio::StreamBuffer stream;
    Radio::StreamBuffer::Metadata publishedMetadata;
    U64 generation = 0;

    Tools::Snapshot<F32> bufferFill{0.0f};
    Tools::Snapshot<U64> deviceOverflows{0};
    Tools::Snapshot<std::pair<F32, F32>> throughput{{0.0f, 0.0f}};
    Tools::Snapshot<SoapyDeviceInfo> deviceInfo;

    Result soapyThreadLoop();
    Result emitFrame();

 private:
    struct Tuning {
        F32 frequency = 0.0f;
        F32 sampleRate = 0.0f;
        bool automaticGain = true;
        F32 gain = 0.0f;
        std::string gainMode = "total";
        Parser::Map gains;
        F64 bandwidth = 0.0;
    };

    SoapyReceiver receiverDevice;
    Radio::StreamAccess access;
    Tuning tuned;

    Result allocateBuffers();
    Result configureDevice();
    Result applyTuning(const Tuning& desired, bool force = false);
    Result updateDeviceInfo(bool reset);
    void resetStream(F64 appliedFrequency, F64 appliedSampleRate);
    Result publishMetadata(const Radio::StreamBuffer::Metadata& metadata);
    Result startReceiver();
    void stopReceiver();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH
