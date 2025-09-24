#ifndef JETSTREAM_BLOCK_THROTTLE_BASE_HH
#define JETSTREAM_BLOCK_THROTTLE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/throttle.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Throttle : public Block {
 public:
    // Configuration

    struct Config {
        U64 intervalMs = 100;

        JST_SERDES(intervalMs);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        mem2::Tensor buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "throttle";
    }

    std::string name() const {
        return "Throttle";
    }

    std::string summary() const {
        return "Throttles tensor data to a specified interval.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Throttles the flow of tensor data by only allowing data to pass through at specified time intervals. The interval is specified in milliseconds and accounts for execution time to maintain accurate timing.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            throttle, "throttle", {
                .intervalMs = config.intervalMs,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, throttle->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (throttle) {
            JST_CHECK(instance().eraseModule(throttle->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Interval (ms)");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 intervalMs = static_cast<F32>(config.intervalMs);
        if (ImGui::InputFloat("##ThrottleInterval", &intervalMs, 1.0f, 10.0f, "%.0f")) {
            if (intervalMs >= 1.0f && intervalMs <= 10000.0f) {
                config.intervalMs = static_cast<U64>(intervalMs);
                throttle->intervalMs(config.intervalMs);
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Throttle<D, IT>> throttle;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Throttle, is_specialized<Jetstream::Throttle<D, IT>>::value &&
                           std::is_same<OT, void>::value)

#endif
