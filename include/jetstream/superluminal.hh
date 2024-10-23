#ifndef JETSTREAM_SUPERLUMINAL_HH
#define JETSTREAM_SUPERLUMINAL_HH

#include <thread>
#include <variant>

#include "jetstream/base.hh"
#include "jetstream/types.hh"

namespace Jetstream {

class Superluminal {
 public:
    enum class Type {
        Line,
        Heat,
        Scatter,
        Waterfall,
    };

    enum class Domain {
        Time,
        Frequency,
    };

    enum class Operation {
        Real,
        Imaginary,
        Amplitude,
        Phase,
    };

    // TODO: Add support for more data types.
    // TODO: Add support for more devices.
    typedef std::variant<Tensor<Device::CPU, CF32>> VariantBufferType;
    typedef std::vector<std::vector<U8>> Mosaic;

    struct PlotConfig {
        VariantBufferType buffer;
        Type type;
        I32 batchAxis = -1;
        I32 channelAxis = -1;
        I32 channelIndex = -1;
        Domain source = Domain::Time;
        Domain display = Domain::Time;
        Operation operation = Operation::Amplitude;
        std::map<std::string, std::variant<std::string, U64, F32>> options;
    };

    struct InstanceConfig {
        InstanceConfig() {}
        // TODO: Add headless option.
        // TODO: Add preferred renderer option.
        U64 deviceId = 0;
        F32 interfaceScale = 1.0f;
        Extent2D<U64> interfaceSize = {1280, 720};
        std::string windowTitle = "Superluminal";
        bool headless = false;
        std::string endpoint = "0.0.0.0:5002";
    };

    static Result Initialize(const InstanceConfig& config = {}) {
        return GetInstance()->initialize(config);
    }

    static Result Terminate() {
        return GetInstance()->terminate();
    }

    static Result Start() {
        return GetInstance()->start();
    }

    static Result Stop() {
        return GetInstance()->stop();
    }

    static bool Presenting() {
        return GetInstance()->presenting();
    }

    static Result Update(const std::string& name = {}) {
        return GetInstance()->update(name);
    }

    static Result Block() {
        return GetInstance()->block();
    }

    static Result PollEvents(const bool& wait = true) {
        return GetInstance()->pollEvents(wait);
    }

    static Result RealtimeLoop(auto lambda) {
        JST_CHECK(Superluminal::Start());

        bool running = true;
        auto child = std::thread(lambda, std::ref(running));

        Superluminal::Block();

        running = false;

        if (child.joinable()) {
            child.join();
        }

        JST_CHECK(Superluminal::Stop());
        JST_CHECK(Superluminal::Terminate());

        return Result::SUCCESS;
    }

    static Result Show() {
        JST_CHECK(Superluminal::Start());
        JST_CHECK(Superluminal::Update());
        JST_CHECK(Superluminal::Block());
        JST_CHECK(Superluminal::Stop());
        JST_CHECK(Superluminal::Terminate());

        return Result::SUCCESS;
    }

    static Result Plot(const std::string& name, const Mosaic& mosaic, const PlotConfig& config) {
        return GetInstance()->plot(name, mosaic, config);
    }

 private:
    Superluminal();
    ~Superluminal();

    static Superluminal* GetInstance();

    struct Impl;
    std::unique_ptr<Impl> impl;

    Result initialize(const InstanceConfig& config = {});
    Result terminate();

    Result start();
    Result stop();

    bool presenting();
    Result update(const std::string& name = {});

    Result block();
    Result pollEvents(const bool& wait = true);
    Result blockAndLoop();

    Result plot(const std::string& name, const Mosaic& mosaic, const PlotConfig& config);
};

}  // namespace Jetstream

#endif  // JETSTREAM_SUPERLUMINAL_HH