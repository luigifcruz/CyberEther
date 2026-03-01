#include <jetstream/domains/dsp/adsb/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/adsb/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct AdsbImpl : public Block::Impl, public DynamicConfig<Blocks::Adsb> {
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Adsb> adsbConfig = std::make_shared<Modules::Adsb>();
    Modules::AdsbImpl* moduleImpl = nullptr;
};

Result AdsbImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Input",
                                   "Raw CF32 IQ samples (1090 MHz, "
                                   "2 MHz sample rate)."));

    JST_CHECK(defineInterfaceMetric("aircraftTable",
                                    "Aircraft",
                                    "Tracked aircraft from decoded "
                                    "ADS-B frames.",
                                    "table",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("No data.");
            }
            return moduleImpl->getAircraftTable();
        }));

    return Result::SUCCESS;
}

Result AdsbImpl::create() {
    JST_CHECK(moduleCreate("adsb", adsbConfig, {
        {"signal", inputs().at("signal")}
    }));

    moduleImpl = moduleHandle("adsb")->getImpl<Modules::AdsbImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(AdsbImpl);

}  // namespace Jetstream::Blocks
