#ifndef JETSTREAM_SUPERLUMINAL_HH
#define JETSTREAM_SUPERLUMINAL_HH

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <thread>
#include <variant>

#include "jetstream/base.hh"
#include "jetstream/types.hh"
#include "jetstream/logger.hh"

namespace Jetstream {

class Superluminal {
 public:
    enum class Type {
        Line,
        Heat,
        Scatter,
        Waterfall,
        Interface,
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

    typedef std::vector<std::vector<U8>> Mosaic;

    struct PlotConfig {
        Tensor buffer;
        Type type;
        I32 batchAxis = -1;
        I32 channelAxis = -1;
        I32 channelIndex = -1;
        Domain source = Domain::Time;
        Domain display = Domain::Time;
        Operation operation = Operation::Amplitude;
        std::map<std::string, std::variant<std::string, I32, F32>> options;
    };

    struct InstanceConfig {
        InstanceConfig() {}
        // TODO: Add preferred renderer option.
        U64 deviceId = 0;
        F32 interfaceScale = 1.0f;
        Extent2D<U64> interfaceSize = {1280, 720};
        std::string windowTitle = "Superluminal";
        bool remote = false;
        DeviceType preferredDevice = DeviceType::CPU;
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

        std::atomic<bool> running = true;
        auto child = std::thread([&running]() {
            while (running.load() && Superluminal::Presenting()) {
                Superluminal::Update();
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
        });

        JST_CHECK(Superluminal::Block());

        running.store(false);
        if (child.joinable()) {
            child.join();
        }

        JST_CHECK(Superluminal::Stop());
        JST_CHECK(Superluminal::Terminate());

        return Result::SUCCESS;
    }

    static Result Plot(const std::string& name, const Mosaic& mosaic, const PlotConfig& config) {
        return GetInstance()->plot(name, mosaic, config);
    }

    static Result Interface(const std::string& name, const Mosaic& mosaic, const std::function<void()>& callback) {
        return GetInstance()->interface(name, mosaic, callback);
    }

    static Result Box(const std::string& title, const Mosaic& mosaic, const std::function<void()>& callback) {
        return GetInstance()->box(title, mosaic, callback);
    }

    static Result Text(const std::string& format, auto&&... args) {
        return GetInstance()->text(jst::fmt::format(jst::fmt::runtime(format), std::forward<decltype(args)>(args)...));
    }

    static Result Slider(const std::string& label, F32 min, F32 max, F32& value) {
        return GetInstance()->slider(label, min, max, value);
    }

    static Result Slider(const std::string& label, F32 min, F32 max, U64& value) {
        F32 fValue = static_cast<F32>(value);
        auto result = GetInstance()->slider(label, min, max, fValue);
        value = static_cast<U64>(fValue);
        return result;
    }

    static Result Markdown(const std::string& content) {
        return GetInstance()->markdown(content);
    }

    static Mosaic MosaicLayout(U8 matrixHeight, U8 matrixWidth,
                               U8 panelHeight, U8 panelWidth,
                               U8 offsetX, U8 offsetY);

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
    Result interface(const std::string& name, const Mosaic& mosaic, const std::function<void()>& callback);
    Result box(const std::string& title, const Mosaic& mosaic, const std::function<void()>& callback);
    Result text(const std::string& content);
    Result slider(const std::string& label, F32 min, F32 max, F32& value);
    Result markdown(const std::string& content);
};

}  // namespace Jetstream

#endif  // JETSTREAM_SUPERLUMINAL_HH
