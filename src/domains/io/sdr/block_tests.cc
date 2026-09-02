#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/io/sdr/block.hh"
#include "jetstream/registry.hh"
#include "scripted_radio.hh"

using namespace Jetstream;

TEST_CASE("SDR block keeps the radio interface defaults",
          "[modules][io][sdr][block]") {
    Blocks::Sdr config;
    REQUIRE(config.deviceString.empty());
    REQUIRE(config.antenna.empty());
    REQUIRE(config.channels == 1);
    REQUIRE(config.manualGains.empty());
    REQUIRE(config.frequency == 96.9e6);
    REQUIRE(config.frequencyStep == 1000000.0);
    REQUIRE(config.sampleRate == 2.0e6f);
    REQUIRE(config.automaticGain);
    REQUIRE_FALSE(config.biasTee);
    REQUIRE(config.numberOfBatches == 8);
    REQUIRE(config.numberOfTimeSamples == 8192);
    REQUIRE(config.bufferMultiplier == 4);
}

TEST_CASE("SDR block selections and settings round-trip through saved configuration",
          "[modules][io][sdr][block]") {
    Blocks::Sdr config;
    config.deviceString = "driver=hackrf,serial=0000000000000000a06063c8234e925f";
    config.antenna = "LNAW";
    config.biasTee = true;
    config.webUsbRefresh = 3;
    config.channels = 2;
    config.manualGains = {30.0f, 36.0f};
    config.frequency = 433.92e6;

    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE_FALSE(serialized.contains("modulePath"));
    REQUIRE_FALSE(serialized.contains("streamString"));
    REQUIRE(serialized.contains("channels"));

    Blocks::Sdr restored;
    REQUIRE(restored.deserialize(serialized) == Result::SUCCESS);
    REQUIRE(restored.deviceString == config.deviceString);
    REQUIRE(restored.antenna == config.antenna);
    REQUIRE(restored.biasTee);
    REQUIRE(restored.webUsbRefresh == 3);
    REQUIRE(restored.channels == 2);
    REQUIRE(restored.manualGains == config.manualGains);
    REQUIRE(restored.frequency == config.frequency);
    REQUIRE(restored.hash() == config.hash());
}

TEST_CASE("SDR block keeps tuning values F32 cannot hold through saved YAML",
          "[modules][io][sdr][block]") {
    constexpr F64 Frequency = 2750.0e6;
    REQUIRE(static_cast<F64>(static_cast<F32>(Frequency)) != Frequency);
    Blocks::Sdr config;
    config.frequency = Frequency;
    config.frequencyStep = 12.5e3;

    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE(serialized.at("frequency").type() == typeid(F64));
    REQUIRE(serialized.at("frequencyStep").type() == typeid(F64));

    std::string yaml;
    REQUIRE(Parser::YamlEncode(serialized, yaml) == Result::SUCCESS);
    Parser::Map decoded;
    REQUIRE(Parser::YamlDecode(yaml, decoded) == Result::SUCCESS);

    Blocks::Sdr restored;
    REQUIRE(restored.deserialize(decoded) == Result::SUCCESS);
    REQUIRE(restored.frequency == Frequency);
    REQUIRE(restored.frequencyStep == 12.5e3);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "SDR block exposes the radio controls and graphical metrics",
                 "[modules][io][sdr][block]") {
    REQUIRE_FALSE(Registry::ListAvailableModules("sdr").empty());

    Parser::Map config;
    config["frequencyStep"] = 0.0f;
    REQUIRE(flowgraph->blockCreate("bad_step", "sdr", config, {}) == Result::SUCCESS);
    const auto bad = viewBlock("bad_step");
    REQUIRE(bad.state == Block::State::Errored);
    REQUIRE(bad.diagnostic.find("[BLOCK_SDR]") != std::string::npos);

    REQUIRE(flowgraph->blockCreate("radio", "sdr", {}, {}) == Result::SUCCESS);
    const auto block = viewBlock("radio");
    REQUIRE(block.state == Block::State::Incomplete);
    REQUIRE(block.interfaceOutputs.size() == 1);
    REQUIRE(block.interfaceOutputs.front().name == "signal");
    for (const auto* name : {"deviceString", "antenna", "frequency", "sampleRate",
                             "automaticGain", "biasTee", "numberOfBatches",
                             "numberOfTimeSamples", "bufferMultiplier"}) {
        INFO(name);
        REQUIRE(std::any_of(block.interfaceConfigs.begin(), block.interfaceConfigs.end(),
                            [&](const auto& field) { return field.name == name; }));
    }

    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("radio", metrics) == Result::SUCCESS);
    const std::vector<std::pair<std::string, Parser::Map>> formats{
        {"bufferFill", {{"type", "progressbar"}, {"warning_above", 0.5f}, {"error_above", 0.75f}}},
        {"bufferLoss", {{"type", "progressbar"}, {"warning_above", 0.0f}, {"error_above", 0.01f}}},
        {"throughput", {{"type", "progressbar"}, {"warning_below", 0.95f}, {"error_below", 0.8f}}},
    };
    for (const auto& [name, format] : formats) {
        INFO(name);
        const auto metric = std::find_if(metrics.begin(), metrics.end(),
                                         [name = name](const auto& entry) { return entry.name == name; });
        REQUIRE(metric != metrics.end());
        REQUIRE(metric->format == format);
    }

    const auto throughput = std::find_if(metrics.begin(), metrics.end(),
                                         [](const auto& entry) { return entry.name == "throughput"; });
    REQUIRE(std::isnan(std::any_cast<std::pair<std::string, F32>>(throughput->value).second));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Built-in radio examples load with the direct SDR blocks",
                 "[modules][io][sdr][block][examples]") {
    const std::vector<std::pair<std::string, std::string>> examples{
        {"spectrum-analyzer", "sdr"},
    };
    for (const auto& [key, source] : examples) {
        DYNAMIC_SECTION("Example: " << key) {
            const auto registrations = Registry::ListAvailableFlowgraphs(key);
            REQUIRE(registrations.size() == 1);
            const auto& content = registrations.front().content;
            REQUIRE(content.find("soapy") == std::string::npos);
            REQUIRE(flowgraph->importFromBlob({content.begin(), content.end()}) == Result::SUCCESS);

            // No radio is selected, so the source waits and nothing fails.
            const auto radio = viewBlock(source);
            REQUIRE(radio.state == Block::State::Incomplete);
            std::vector<std::string> names;
            REQUIRE(flowgraph->view().keys(names) == Result::SUCCESS);
            for (const auto& name : names) {
                INFO(name << ": " << viewBlock(name).diagnostic);
                REQUIRE(viewBlock(name).state != Block::State::Errored);
            }
        }
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "SDR block loads saved SoapySDR blocks",
                 "[modules][io][sdr][block]") {
    const TestRadio::ScriptedDrivers drivers;
    const std::string yaml = R"(---
version: 2
graph:
  - name: remote
    module: soapy
    config:
      deviceString: 'driver=uhd, serial=3141592'
      streamString: 'bufflen=8192'
      modulePath: '/opt/soapy/modules'
      antenna: RX2
  - name: unnamed
    module: soapy
    config:
      deviceString: 'driver=rtltcp,rtltcp=127.0.0.1:1234'
  - name: bare
    module: soapy
  - name: indexed
    module: soapy
    config:
      deviceString: 'driver=rtlsdr, index=1'
      webUsbRefresh: 3
  - name: indexedserial
    module: soapy
    config:
      deviceString: 'driver=rtlsdr,index=2,serial=00000001'
  - name: lime
    module: soapy
    config:
      deviceString: 'driver=lime,serial=1D4C2F'
      automaticGain: false
      manualGain: 30
      antenna: NONE
  - name: limedefault
    module: soapy
    config:
      deviceString: 'driver=lime,serial=1D4C30'
  - name: kept
    module: soapy
    config:
      deviceString: 'driver=uhd,serial=2718281'
      manualGain: 49.5
)";

    const std::vector<char> blob(yaml.begin(), yaml.end());
    REQUIRE(flowgraph->importFromBlob(blob) == Result::SUCCESS);

    for (const auto* name : {"remote", "unnamed", "bare"}) {
        INFO(name);
        REQUIRE(viewBlock(name).type == "sdr");
    }

    Parser::Map remote;
    REQUIRE(flowgraph->blockConfig("remote", remote) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(remote.at("deviceString")) == "driver=soapy,serial=3141592");
    REQUIRE(std::any_cast<std::string>(remote.at("driverArguments")) == "bufflen=8192");
    REQUIRE(std::any_cast<std::string>(remote.at("antenna")) == "RX2");
    REQUIRE(std::any_cast<F32>(remote.at("manualGain")) == Blocks::Sdr{}.manualGain);

    Parser::Map unnamed;
    REQUIRE(flowgraph->blockConfig("unnamed", unnamed) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(unnamed.at("deviceString")).empty());

    Parser::Map bare;
    REQUIRE(flowgraph->blockConfig("bare", bare) == Result::SUCCESS);
    REQUIRE(std::any_cast<F32>(bare.at("manualGain")) == Blocks::Sdr{}.manualGain);

#ifdef JETSTREAM_LOADER_RTLSDR_AVAILABLE
    for (const auto& [name, index] : std::vector<std::pair<std::string, U32>>{
             {"indexed", 1}, {"indexedserial", 2}}) {
        Parser::Map indexed;
        REQUIRE(viewBlock(name).type == "sdr");
        REQUIRE(flowgraph->blockConfig(name, indexed) == Result::SUCCESS);
        REQUIRE(std::any_cast<std::string>(indexed.at("deviceString")) ==
                jst::fmt::format("driver=rtlsdr,index={}", index));
        if (name == "indexed") {
            REQUIRE(std::any_cast<U64>(indexed.at("webUsbRefresh")) == 3);
        }
    }
#endif

#ifdef JETSTREAM_LOADER_LIMESUITE_AVAILABLE
    Parser::Map lime;
    REQUIRE(flowgraph->blockConfig("lime", lime) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(lime.at("deviceString")) == "driver=limesdr,serial=1D4C2F");
    REQUIRE(std::any_cast<F32>(lime.at("manualGain")) == 42.0f);
    REQUIRE(std::any_cast<std::string>(lime.at("antenna")).empty());

    Parser::Map limeDefault;
    REQUIRE(flowgraph->blockConfig("limedefault", limeDefault) == Result::SUCCESS);
    REQUIRE(std::any_cast<F32>(limeDefault.at("manualGain")) == 44.0f);
#endif

    Parser::Map kept;
    REQUIRE(flowgraph->blockConfig("kept", kept) == Result::SUCCESS);
    REQUIRE(std::any_cast<F32>(kept.at("manualGain")) == 49.5f);
}

TEST_CASE("SDR block keeps single RX defaults for older configurations",
          "[modules][sdr][multi-channel][block]") {
    Blocks::Sdr config;
    REQUIRE(config.deserialize({{"frequency", F32{100.0e6}}}) == Result::SUCCESS);
    REQUIRE(config.channels == 1);
    REQUIRE(config.manualGains.empty());
}
