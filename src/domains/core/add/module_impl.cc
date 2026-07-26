#include "module_impl.hh"

#include <algorithm>

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result AddImpl::validate() {
    validatedA = Tensor();
    validatedB = Tensor();
    validatedOutputShape.clear();
    validatedOutputSizeBytes = 0;

    if (!inputs().contains("a") || !inputs().contains("b")) {
        return Result::SUCCESS;
    }

    const Tensor& tensorA = inputs().at("a").tensor;
    const Tensor& tensorB = inputs().at("b").tensor;

    // The framework owns malformed and empty input rejection after define().
    if (!tensorA.validShape() || !tensorB.validShape() ||
        tensorA.size() == 0 || tensorB.size() == 0) {
        return Result::SUCCESS;
    }

    const Shape& shapeA = tensorA.shape();
    const Shape& shapeB = tensorB.shape();
    const U64 rankA = shapeA.size();
    const U64 rankB = shapeB.size();
    const U64 maxRank = std::max(rankA, rankB);

    Shape outputShape(maxRank == 0 ? 1 : maxRank, 1);
    for (U64 i = 0; i < maxRank; ++i) {
        const U64 dimA = rankA > i ? shapeA[rankA - 1 - i] : 1;
        const U64 dimB = rankB > i ? shapeB[rankB - 1 - i] : 1;

        if (dimA != dimB && dimA != 1 && dimB != 1) {
            JST_ERROR("[MODULE_ADD] Input shapes {} and {} are not broadcastable.",
                      shapeA, shapeB);
            return Result::ERROR;
        }

        outputShape[outputShape.size() - 1 - i] = std::max(dimA, dimB);
    }

    U64 outputSize = 1;
    for (const U64 dim : outputShape) {
        if (!detail::CheckedMultiply(outputSize, dim, outputSize)) {
            JST_ERROR("[MODULE_ADD] Broadcast output shape {} exceeds the supported layout range.",
                      outputShape);
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputSize,
                                 static_cast<U64>(DataTypeSize(tensorA.dtype())),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_ADD] Broadcast output shape {} exceeds the supported byte range.",
                  outputShape);
        return Result::ERROR;
    }

    Tensor broadcastA = tensorA.clone();
    Tensor broadcastB = tensorB.clone();
    if (broadcastA.broadcastTo(outputShape) != Result::SUCCESS ||
        broadcastB.broadcastTo(outputShape) != Result::SUCCESS) {
        JST_ERROR("[MODULE_ADD] Failed to construct validated broadcast views.");
        return Result::ERROR;
    }

    validatedA = std::move(broadcastA);
    validatedB = std::move(broadcastB);
    validatedOutputShape = std::move(outputShape);
    validatedOutputSizeBytes = outputSizeBytes;
    return Result::SUCCESS;
}

Result AddImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceOutput("sum"));

    JST_CHECK(defineInterfaceInput("a"));
    JST_CHECK(defineInterfaceInput("b"));

    return Result::SUCCESS;
}

Result AddImpl::create() {
    JST_TRACE("[MODULE_ADD] Output shape {}.", validatedOutputShape);

    const DeviceType device = validatedA.device();
    const DataType dtype = validatedA.dtype();

    a = validatedA;
    b = validatedB;

    JST_CHECK(c.create(device, dtype, validatedOutputShape));

    c.propagateAttributes(a);

    {
        Tensor inputA = a;
        Tensor inputB = b;

        c.setDerivedAttribute("sampleRate", [inputA, inputB]() -> std::any {
            const auto srA = inputA.hasAttribute("sampleRate") ? std::any_cast<F32>(inputA.attribute("sampleRate")) : 0.0f;
            const auto srB = inputB.hasAttribute("sampleRate") ? std::any_cast<F32>(inputB.attribute("sampleRate")) : 0.0f;
            if (srA == srB || srB == 0.0f) {
                return std::any(srA);
            } else if (srA == 0.0f) {
                return std::any(srB);
            }
            return std::any(srA);
        });

        c.setDerivedAttribute("frequency", [inputA, inputB]() -> std::any {
            const auto fA = inputA.hasAttribute("frequency") ? std::any_cast<F32>(inputA.attribute("frequency")) : 0.0f;
            const auto fB = inputB.hasAttribute("frequency") ? std::any_cast<F32>(inputB.attribute("frequency")) : 0.0f;
            if (fA == fB || fB == 0.0f) {
                return std::any(fA);
            } else if (fA == 0.0f) {
                return std::any(fB);
            }
            return std::any(fA);
        });
    }

    outputs()["sum"].produced(name(), "sum", c);

    return Result::SUCCESS;
}

Result AddImpl::destroy() {
    return Result::SUCCESS;
}

Result AddImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
