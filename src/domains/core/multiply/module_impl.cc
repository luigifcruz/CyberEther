#include "module_impl.hh"

#include <algorithm>

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result MultiplyImpl::validate() {
    validatedA = Tensor();
    validatedB = Tensor();
    validatedOutputShape.clear();
    validatedOutputElementCount = 0;
    validatedOutputSizeBytes = 0;

    if (!inputs().contains("a") || !inputs().contains("b")) {
        return Result::SUCCESS;
    }

    const Tensor& tensorA = inputs().at("a").tensor;
    const Tensor& tensorB = inputs().at("b").tensor;
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
            JST_ERROR("[MODULE_MULTIPLY] Input shapes {} and {} are not broadcastable.",
                      shapeA, shapeB);
            return Result::ERROR;
        }

        outputShape[outputShape.size() - 1 - i] = std::max(dimA, dimB);
    }

    Tensor mappedAxes(DeviceType::CPU, DataType::F32,
                      Shape(outputShape.size(), 1));
    JST_CHECK(MergeBroadcastSignalAxes(tensorA, tensorB, mappedAxes));

    U64 outputElementCount = 1;
    for (const U64 dimension : outputShape) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[MODULE_MULTIPLY] Broadcast output exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(tensorA.dtype())),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_MULTIPLY] Broadcast output exceeds the supported byte range.");
        return Result::ERROR;
    }

    Tensor broadcastA = tensorA.clone();
    Tensor broadcastB = tensorB.clone();
    if (broadcastA.broadcastTo(outputShape) != Result::SUCCESS ||
        broadcastB.broadcastTo(outputShape) != Result::SUCCESS) {
        JST_ERROR("[MODULE_MULTIPLY] Failed to construct validated broadcast views.");
        return Result::ERROR;
    }

    validatedA = std::move(broadcastA);
    validatedB = std::move(broadcastB);
    validatedOutputShape = std::move(outputShape);
    validatedOutputElementCount = outputElementCount;
    validatedOutputSizeBytes = outputSizeBytes;
    return Result::SUCCESS;
}

Result MultiplyImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceOutput("product"));

    JST_CHECK(defineInterfaceInput("a"));
    JST_CHECK(defineInterfaceInput("b"));

    return Result::SUCCESS;
}

Result MultiplyImpl::create() {
    a = validatedA;
    b = validatedB;

    JST_CHECK(c.create(a.device(), a.dtype(), validatedOutputShape));

    JST_CHECK(c.propagateAttributes(a));
    JST_CHECK(MergeBroadcastSignalAxes(inputs().at("a").tensor,
                                       inputs().at("b").tensor, c));

    {
        Tensor inputA = a;
        Tensor inputB = b;

        if (inputA.hasAttribute("sampleRate") ||
            inputB.hasAttribute("sampleRate")) {
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
        }

        if (inputA.hasAttribute("frequency") ||
            inputB.hasAttribute("frequency")) {
            c.setDerivedAttribute("frequency", [inputA, inputB]() -> std::any {
                const auto fA = inputA.hasAttribute("frequency") ? std::any_cast<F32>(inputA.attribute("frequency")) : 0.0f;
                const auto fB = inputB.hasAttribute("frequency") ? std::any_cast<F32>(inputB.attribute("frequency")) : 0.0f;
                return std::any(fA + fB);
            });
        }
    }

    outputs()["product"].produced(name(), "product", c);

    return Result::SUCCESS;
}

Result MultiplyImpl::destroy() {
    return Result::SUCCESS;
}

Result MultiplyImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
