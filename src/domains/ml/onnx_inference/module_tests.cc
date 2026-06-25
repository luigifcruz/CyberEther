#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include "jetstream/domains/ml/onnx_inference/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"
#include "onnx_model_helpers.hh"

using namespace Jetstream;

namespace {

template<typename T>
void PopulateTensor(TypedTensor<T>& tensor, const std::array<T, 4>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        tensor.at(static_cast<U64>(i)) = values[i];
    }
}

template<typename T>
void ExpectIdentityRoundTrip(const std::string& onnxTypeName,
                             const std::array<T, 4>& inputValues) {
    const auto modelPath = Jetstream::Tests::CreateIdentityOnnxModel(onnxTypeName,
                                                                     "onnx-" + onnxTypeName);
    const auto implementations = Registry::ListAvailableModules("onnx_inference");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Type: " << onnxTypeName
                        << " Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("onnx_inference", impl.device, impl.runtime, impl.provider);

            Modules::OnnxInference config;
            config.modelPath = modelPath.string();
            config.inputNames = {"input_0"};
            config.outputNames = {"output_0"};
            config.executionProvider = "cpu";
            ctx.setConfig(config);

            auto input = ctx.createTensor<T>({4});
            PopulateTensor(input, inputValues);
            ctx.setInput("input_0", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& output = ctx.output("output_0");
            REQUIRE(output.dtype() == TypeToDataType<T>());
            REQUIRE(output.shape() == std::vector<U64>{4});

            const auto* outData = output.data<T>();
            for (size_t i = 0; i < inputValues.size(); ++i) {
                REQUIRE(outData[i] == inputValues[i]);
            }
        }
    }
}

}  // namespace

TEST_CASE("ONNX inference module - F32 regression", "[modules][onnx_inference][F32]") {
    ExpectIdentityRoundTrip<F32>("FLOAT", std::array<F32, 4>{1.0f, -2.5f, 3.25f, 0.0f});
}

TEST_CASE("ONNX inference module - INT8 round trip", "[modules][onnx_inference][I8]") {
    ExpectIdentityRoundTrip<I8>("INT8", std::array<I8, 4>{1, -2, 3, 0});
}

TEST_CASE("ONNX inference module - UINT8 round trip", "[modules][onnx_inference][U8]") {
    ExpectIdentityRoundTrip<U8>("UINT8", std::array<U8, 4>{1, 2, 3, 0});
}

TEST_CASE("ONNX inference module - rejects mismatched input dtype",
          "[modules][onnx_inference][error][mixed]") {
    const auto modelPath = Jetstream::Tests::CreateIdentityOnnxModel("INT8", "onnx-int8-mismatch");
    const auto implementations = Registry::ListAvailableModules("onnx_inference");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("onnx_inference", impl.device, impl.runtime, impl.provider);

            Modules::OnnxInference config;
            config.modelPath = modelPath.string();
            config.inputNames = {"input_0"};
            config.outputNames = {"output_0"};
            config.executionProvider = "cpu";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            PopulateTensor(input, std::array<F32, 4>{1.0f, 2.0f, 3.0f, 4.0f});
            ctx.setInput("input_0", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
