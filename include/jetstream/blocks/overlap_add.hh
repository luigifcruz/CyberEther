#ifndef JETSTREAM_BLOCK_OVERLAP_ADD_BASE_HH
#define JETSTREAM_BLOCK_OVERLAP_ADD_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/overlap_add.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class OverlapAdd : public Block {
 public:
    // Configuration

    struct Config {
        U64 axis = 0;

        JST_SERDES(axis);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;
        Tensor<D, IT> overlap;

        JST_SERDES(buffer, overlap);
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

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "overlap-add";
    }

    std::string name() const {
        return "Overlap Add";
    }

    std::string summary() const {
        return "Sums overlap with buffer.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Sums the overlap data with the buffer data along the specified axis.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            overlap_add, "overlap_add", {
                .axis = config.axis,
            }, {
                .buffer = input.buffer,
                .overlap = input.overlap,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, overlap_add->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(overlap_add->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Axis");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 axis = config.axis;
        if (ImGui::InputFloat("##overlap-axis", &axis, 1.0f, 1.0f, "%.0f")) {
            if (axis >= 0 and axis < input.buffer.rank()) {
                config.axis = static_cast<U64>(axis);

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
    std::shared_ptr<Jetstream::OverlapAdd<D, IT>> overlap_add;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(OverlapAdd, is_specialized<Jetstream::OverlapAdd<D, IT>>::value &&
                             std::is_same<OT, void>::value)

#endif
