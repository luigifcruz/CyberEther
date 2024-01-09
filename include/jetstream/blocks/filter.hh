#ifndef JETSTREAM_BLOCK_FILTER_BASE_HH
#define JETSTREAM_BLOCK_FILTER_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/blocks/filter_taps.hh"
#include "jetstream/blocks/filter_engine.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Filter : public Block {
 public:
    // Configuration

    struct Config {
        std::vector<F32> center = {0.0e6f};
        F32 sampleRate = 2.0e6f;
        F32 bandwidth = 1.0e6f;
        U64 taps = 101;

        JST_SERDES(center, sampleRate, bandwidth, taps);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputCoeffs() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "filter";
    }

    std::string name() const {
        return "Filter";
    }

    std::string summary() const {
        return "Filters input signal with a bandpass filter.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "The Filter module creates Finite Impulse Response (FIR) bandpass filter coefficients (taps) based on specified frequency parameters. These taps can be used to filter input data, attenuating or amplifying certain frequency components.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addInternalBlock(
            taps, "taps", {
                .center = config.center,
                .sampleRate = config.sampleRate,
                .bandwidth = config.bandwidth,
                .taps = config.taps,
            }, {},
            locale()
        ));

        JST_CHECK(instance().addInternalBlock(
            engine, "engine", {}, {
                .signal = input.buffer,
                .filter = taps->getOutputCoeffs(),
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, engine->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseInternalBlock(engine));
        JST_CHECK(instance().eraseInternalBlock(taps));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 sampleRate = config.sampleRate / JST_MHZ;
        if (ImGui::InputFloat("##filter-sample-rate", &sampleRate, 1.0f, 1.0f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.sampleRate = sampleRate * JST_MHZ;
            
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Bandwidth");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 bandwidth = config.bandwidth / JST_MHZ;
        if (ImGui::InputFloat("##filter-bandwidth", &bandwidth, 1.0f, 1.0f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.bandwidth = bandwidth * JST_MHZ;
            
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Taps");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 taps = config.taps;
        if (ImGui::InputFloat("##filter-taps", &taps, 2.0f, 2.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.taps = static_cast<U64>(taps);
            
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextFormatted("Heads");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 heads = config.center.size();
        if (ImGui::InputFloat("##filter-heads", &heads, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (heads != config.center.size()) {
                config.center.resize(heads);
    
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }

        for (U64 i = 0; i < config.center.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextFormatted("Center #{:02}", i);
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            const std::string id = fmt::format("##filter-center-{}", i);
            F32 center = config.center[i] / JST_MHZ;
            if (ImGui::InputFloat(id.c_str(), &center, 1.0f, 1.0f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
                config.center[i] = center * JST_MHZ;
                
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Blocks::FilterTaps<D, IT, OT>> taps;
    std::shared_ptr<Blocks::FilterEngine<D, IT, OT>> engine;

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Filter, is_specialized<Blocks::FilterTaps<D, IT, OT>>::value &&
                         is_specialized<Blocks::FilterEngine<D, IT, OT>>::value &&
                         std::is_same<OT, void>::value)

#endif
