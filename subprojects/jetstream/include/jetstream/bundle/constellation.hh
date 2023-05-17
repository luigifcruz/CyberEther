#ifndef JETSTREAM_BUNDLE_CONSTELLATION_BASE_HH
#define JETSTREAM_BUNDLE_CONSTELLATION_BASE_HH

#include "jetstream/instance.hh"
#include "jetstream/modules/constellation.hh"

namespace Jetstream::Bundle {

template<Device DeviceId>
class ConstellationUI {
 public:
    using Module = Constellation<DeviceId>;

    Result init(Instance& instance, 
                const typename Module::Config& config,
                const typename Module::Input& input) {
        module =  instance.addBlock<Constellation, DeviceId>(config, input);

        return Result::SUCCESS;
    }

    Result draw() {
        ImGui::Begin("Constellation");

        auto [x, y] = ImGui::GetContentRegionAvail();
        auto [width, height] = module->viewSize({(U64)x, (U64)y});
        ImGui::Image(module->getTexture().raw(), ImVec2(width, height));

        ImGui::End();

        return Result::SUCCESS;       
    }

    Result drawControl() {
            
        return Result::SUCCESS;       
    }

    Result drawInfo() {
            
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
