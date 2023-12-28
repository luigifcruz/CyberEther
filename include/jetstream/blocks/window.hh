#ifndef JETSTREAM_BLOCK_WINDOW_BASE_HH
#define JETSTREAM_BLOCK_WINDOW_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/window.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Window : public Block {
 public:
    // Configuration

    struct Config {
        U64 size;

        JST_SERDES(size);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> window;

        JST_SERDES(window);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputWindow() const {
        return this->output.window;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "window";
    }

    std::string name() const {
        return "Window";
    }

    std::string summary() const {
        return "Generated a Butterworth window.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Renerate a Butterworth window of the specified length and order.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().template addModule<Jetstream::Window, D, IT>(
            window, "window", {
                .size = config.size,
            }, {},
            locale().blockId
        ));

        JST_CHECK(Block::LinkOutput("window", output.window, window->getOutputWindow()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(window->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Taps");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 size = config.size;
        if (ImGui::InputFloat("##window-size", &size, 2.0f, 2.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.size = static_cast<U64>(size);

            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Window<D, IT>> window;

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Window, is_specialized<Jetstream::Window<D, IT>>::value &&
                         std::is_same<OT, void>::value)

#endif
