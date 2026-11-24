#include "jetstream/platform.hh"
#include "jetstream/plugin.hh"
#include "jetstream/registry.hh"

namespace {

[[maybe_unused]] const Jetstream::Result fixtureRegistration =
    Jetstream::Registry::QueueStaticRegistration([]() {
        const auto root = Jetstream::Plugin::Path(JETSTREAM_TEST_PLUGIN_ID);
        return Jetstream::Registry::RegisterBlock(JETSTREAM_TEST_PLUGIN_BLOCK,
                                                  "Fixture",
                                                  "Fixture Probe",
                                                  Jetstream::Platform::PathToUtf8(root),
                                                  "",
                                                  []() -> std::shared_ptr<Jetstream::Block> {
                                                      return nullptr;
                                                  });
    });

}  // namespace
