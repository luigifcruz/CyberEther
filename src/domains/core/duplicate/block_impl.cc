#include <jetstream/domains/core/duplicate/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/duplicate/module.hh>

namespace Jetstream::Blocks {

struct DuplicateImpl : public Block::Impl, public DynamicConfig<Blocks::Duplicate> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Duplicate> moduleConfig = std::make_shared<Modules::Duplicate>();
};

Result DuplicateImpl::validate() {
    const auto& config = *candidate();

    if (hostAccessible != config.hostAccessible) {
        return Result::RECREATE;
    }

    if (outputDevice != config.outputDevice) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result DuplicateImpl::configure() {
    moduleConfig->hostAccessible = hostAccessible;
    moduleConfig->outputDevice = outputDevice;

    return Result::SUCCESS;
}

Result DuplicateImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Signal to be duplicated."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Duplicated signal."));

    JST_CHECK(defineInterfaceConfig("outputDevice",
                                    "Output Device",
                                    "Selects the output device for the duplicated buffer.",
                                    "dropdown:none(None),cpu(CPU),cuda(CUDA),metal(Metal),vulkan(Vulkan)"));

    if (StringToDevice(outputDevice) != DeviceType::CPU) {
        JST_CHECK(defineInterfaceConfig("hostAccessible",
                                        "Host Accessible",
                                        "When enabled, the output buffer can be accessed from the CPU.",
                                        "bool"));
    }

    return Result::SUCCESS;
}

Result DuplicateImpl::create() {
    JST_CHECK(moduleCreate("duplicate", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"duplicate", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(DuplicateImpl);

}  // namespace Jetstream::Blocks
