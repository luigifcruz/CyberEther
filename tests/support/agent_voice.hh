#ifndef JETSTREAM_TESTS_SUPPORT_AGENT_VOICE_HH
#define JETSTREAM_TESTS_SUPPORT_AGENT_VOICE_HH

#include "jetstream/agent/live.hh"
#include "jetstream/agent/session.hh"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace AgentTest {

struct SilentAudio final : Jetstream::Agent::Session::Audio {
    SilentAudio() : Audio(24000) {}

    Jetstream::Result start() override {
        return Jetstream::Result::SUCCESS;
    }

    void stop() override {}

    std::vector<Jetstream::I16> capture(Jetstream::U64) override {
        return {};
    }

    void play(const std::string&) override {}

    void flush() override {}

    Jetstream::F32 input() const override {
        return 0.0f;
    }

    Jetstream::F32 output() const override {
        return 0.0f;
    }
};

inline std::unique_ptr<Jetstream::Agent::Session::Audio> Silent(Jetstream::U32) {
    return std::make_unique<SilentAudio>();
}

struct VoiceLink {
    using Live = Jetstream::Agent::Live;

    struct Connection final : Live {
        VoiceLink& link;

        Connection(VoiceLink& link, Config config, Callbacks callbacks)
            : Live(config, callbacks), link(link) {
            std::lock_guard lock(link.mutex);
            link.configs.push_back(std::move(config));
            link.callbacks.push_back(std::move(callbacks));
            link.changed.notify_all();
        }

        Jetstream::Result start() override {
            return Jetstream::Result::SUCCESS;
        }

        void stop() override {}

        void send(const std::vector<Jetstream::I16>&) override {}

        void reply(const ToolOutputs& outputs) override {
            std::lock_guard lock(link.mutex);
            link.outputs.push_back(outputs);
            link.changed.notify_all();
        }
    };

    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Live::Config> configs;
    std::vector<Live::Callbacks> callbacks;
    std::vector<Live::ToolOutputs> outputs;

    Jetstream::Agent::Session::LiveFactory factory() {
        return [this](Live::Config config, Live::Callbacks callbacks) -> std::unique_ptr<Live> {
            return std::make_unique<Connection>(*this, std::move(config), std::move(callbacks));
        };
    }

    std::optional<Live::Callbacks> connection(size_t index) {
        std::unique_lock lock(mutex);
        const bool ready = changed.wait_for(lock, std::chrono::seconds(5), [&] {
            return callbacks.size() > index;
        });
        if (!ready) {
            return std::nullopt;
        }
        return callbacks[index];
    }

    bool waitForOutputs(size_t count) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [&] {
            return outputs.size() >= count;
        });
    }

    size_t connections() {
        std::lock_guard lock(mutex);
        return callbacks.size();
    }
};

}  // namespace AgentTest

#endif  // JETSTREAM_TESTS_SUPPORT_AGENT_VOICE_HH
