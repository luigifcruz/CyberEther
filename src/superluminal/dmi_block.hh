#ifndef JETSTREAM_SUPERLUMINAL_DMI_BLOCK_HH
#define JETSTREAM_SUPERLUMINAL_DMI_BLOCK_HH

#include <memory>
#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class DynamicMemoryImport : public Block {
 public:
    // Configuration

    struct Config {
        Tensor<D, OT> buffer;

        JST_SERDES(buffer);
    };

    // Input

    struct Input {
        JST_SERDES();
    };

    // Output

    struct Output {
        Tensor<D, OT> buffer;

        JST_SERDES(buffer);
    };

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "dynamic-tensor-import";
    }

    std::string name() const {
        return "Dynamic Tensor Import";
    }

    std::string summary() const {
        return "Dynamically imports an external tensor.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Dynamically imports an external tensor.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            dynamicMemoryImport, "source", {
                .buffer = config.buffer,
            }, {},
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, dynamicMemoryImport->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(dynamicMemoryImport->locale()));
        return Result::SUCCESS;
    }

    // Interface

    constexpr bool shouldDrawInfo() const {
        return true;
    }

    void drawInfo() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Shape");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::TextFormatted("{}", output.buffer.shape());
    }

 private:
    JST_DEFINE_IO()

    std::shared_ptr<Jetstream::DynamicMemoryImport<D, OT>> dynamicMemoryImport;
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(DynamicMemoryImport, std::is_same<IT, void>::value &&
                                      is_specialized<Jetstream::DynamicMemoryImport<D, OT>>::value)

#endif  // JETSTREAM_SUPERLUMINAL_DMI_BLOCK_HH