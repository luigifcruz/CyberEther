#include <jetstream/domains/core/throttle/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/core/throttle/module.hh>

namespace Jetstream::Blocks {

struct ThrottleImpl : public Block::Impl, public DynamicConfig<Blocks::Throttle> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Throttle> throttleConfig = std::make_shared<Modules::Throttle>();
};

Result ThrottleImpl::configure() {
    throttleConfig->pacing = pacing;
    throttleConfig->intervalMs = intervalMs;

    return Result::SUCCESS;
}

Result ThrottleImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "Input data to throttle."));

    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Throttled output data (same as input)."));

    JST_CHECK(defineInterfaceConfig("pacing",
                                    "Pacing",
                                    "What sets the time between outputs.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Interval"}, {"value", "interval"}},
                                        Parser::Map{{"label", "Sample Rate"}, {"value", "sampleRate"}},
                                    }}}));

    if (candidate()->pacing == "interval") {
        JST_CHECK(defineInterfaceConfig("intervalMs",
                                        "Interval",
                                        "Time between outputs in milliseconds.",
                                        {{"type", "float"}, {"unit", "ms"}, {"precision", 3}}));
    }

    return Result::SUCCESS;
}

Result ThrottleImpl::create() {
    JST_CHECK(moduleCreate("throttle", throttleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"throttle", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ThrottleImpl, {"throttle"});

}  // namespace Jetstream::Blocks
