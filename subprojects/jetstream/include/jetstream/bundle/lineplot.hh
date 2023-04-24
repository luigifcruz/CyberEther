#ifndef JETSTREAM_BUNDLE_LINEPLOT_BASE_HH
#define JETSTREAM_BUNDLE_LINEPLOT_BASE_HH

#include "jetstream/instance.hh"
#include "jetstream/modules/lineplot.hh"

namespace Jetstream::Bundle {

template<Device DeviceId>
class LineplotUI {
 public:
    using Module = Lineplot<DeviceId>;

    const Result init(Instance& instance,
                      const typename Module::Config& config,
                      const typename Module::Input& input) {
        module = instance.addBlock<Lineplot, DeviceId>(config, input);

        return Result::SUCCESS;
    }

    const Result draw() {
        ImGui::Begin("Lineplot");
        
        auto [x, y] = ImGui::GetContentRegionAvail();
        auto [width, height] = module->viewSize({(U64)x, (U64)y});
        ImGui::Image(module->getTexture().raw(), ImVec2(width, height));

        ImGui::End();

        return Result::SUCCESS;       
    }

    const Result drawControl() {
            
        return Result::SUCCESS;       
    }

    const Result drawInfo() {
            
        return Result::SUCCESS;       
    }

    constexpr Module& get() {
        return *module;
    }

 private:
    std::shared_ptr<Module> module;
};

}  // namespace Jetstream::Bundle

#endif
