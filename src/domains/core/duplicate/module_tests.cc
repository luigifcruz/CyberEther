#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

#include "jetstream/domains/core/duplicate/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

template<typename T>
void expectDuplicateSuccess(const std::string& tag,
                            const std::vector<T>& values) {
    const auto implementations = Registry::ListAvailableModules("duplicate");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Type: " << tag << " Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("duplicate", impl.device, impl.runtime, impl.provider);

            Modules::Duplicate config;
            config.hostAccessible = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<T>({values.size()});
            for (U64 i = 0; i < values.size(); ++i) {
                input.at(i) = values[i];
            }

            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape(0) == values.size());
            REQUIRE(out.dtype() == TypeToDataType<T>());

            for (U64 i = 0; i < values.size(); ++i) {
                REQUIRE(out.at<T>(i) == values[i]);
            }
        }
    }
}

}  // namespace

TEST_CASE("Duplicate Module - F32", "[modules][duplicate][F32]") {
    expectDuplicateSuccess<F32>("F32", {1.0f, -2.0f, 3.0f, -4.0f});
}

TEST_CASE("Duplicate Module - Full dtype coverage", "[modules][duplicate][all]") {
    expectDuplicateSuccess<F64>("F64", {1.0, -2.0, 3.0});
    expectDuplicateSuccess<I8>("I8", {1, -2, 3, -4});
    expectDuplicateSuccess<I16>("I16", {1024, -2048, 4096});
    expectDuplicateSuccess<I32>("I32", {1024, -2048, 4096});
    expectDuplicateSuccess<I64>("I64", {1024, -2048, 4096});
    expectDuplicateSuccess<U8>("U8", {1, 2, 3, 4});
    expectDuplicateSuccess<U16>("U16", {1024, 2048, 4096});
    expectDuplicateSuccess<U32>("U32", {1024, 2048, 4096});
    expectDuplicateSuccess<U64>("U64", {1024, 2048, 4096});
    expectDuplicateSuccess<CF32>("CF32", {{1.0f, 2.0f}, {3.0f, -4.0f}});
    expectDuplicateSuccess<CF64>("CF64", {{1.0, 2.0}, {3.0, -4.0}});
    expectDuplicateSuccess<CI8>("CI8", {{1, 2}, {3, -4}});
    expectDuplicateSuccess<CI16>("CI16", {{1024, -2048}, {4096, 8192}});
    expectDuplicateSuccess<CI32>("CI32", {{1024, -2048}, {4096, 8192}});
    expectDuplicateSuccess<CI64>("CI64", {{1024, -2048}, {4096, 8192}});
    expectDuplicateSuccess<CU8>("CU8", {{1, 2}, {3, 4}});
    expectDuplicateSuccess<CU16>("CU16", {{1024, 2048}, {4096, 8192}});
    expectDuplicateSuccess<CU32>("CU32", {{1024, 2048}, {4096, 8192}});
    expectDuplicateSuccess<CU64>("CU64", {{1024, 2048}, {4096, 8192}});
}
