#include "module_impl.hh"

#include <algorithm>

namespace Jetstream::Modules {

Result AddImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceOutput("sum"));

    JST_CHECK(defineInterfaceInput("a"));
    JST_CHECK(defineInterfaceInput("b"));

    return Result::SUCCESS;
}

Result AddImpl::create() {
    const Tensor& tensorA = inputs().at("a").tensor;
    const Tensor& tensorB = inputs().at("b").tensor;

    const Shape& shapeA = tensorA.shape();
    const Shape& shapeB = tensorB.shape();

    const U64 rankA = shapeA.size();
    const U64 rankB = shapeB.size();

    const U64 minRank = std::min(rankA, rankB);
    const U64 maxRank = std::max(rankA, rankB);

    const U64 padA = maxRank - rankB;
    const U64 padB = maxRank - rankA;

    JST_TRACE("[MODULE_ADD] A rank: {}; B rank: {}.", rankA, rankB);
    JST_TRACE("[MODULE_ADD] Min rank: {}; Max rank: {}.", minRank, maxRank);
    JST_TRACE("[MODULE_ADD] Pad A: {}; Pad B: {}.", padA, padB);

    for (U64 i = 0; i < minRank; ++i) {
        const U64 dimA = shapeA[padA + i];
        const U64 dimB = shapeB[padB + i];

        JST_TRACE("[MODULE_ADD] Checking rank {} -> {} vs {}.", i, dimA, dimB);

        if (dimA != dimB && dimA != 1 && dimB != 1) {
            JST_ERROR("[MODULE_ADD] Input shapes {} and {} are not broadcastable.", shapeA, shapeB);
            return Result::ERROR;
        }
    }

    Shape outputShape;
    if (maxRank == 0) {
        outputShape.push_back(1);
    } else {
        outputShape.resize(maxRank);

        for (U64 i = 0; i < maxRank; ++i) {
            const U64 indexA = rankA > i ? shapeA[rankA - 1 - i] : 1;
            const U64 indexB = rankB > i ? shapeB[rankB - 1 - i] : 1;

            if (indexA == 0 || indexB == 0) {
                JST_ERROR("[MODULE_ADD] Tensor dimensions cannot be zero ({} vs {}).",
                          indexA, indexB);
                return Result::ERROR;
            }

            outputShape[maxRank - 1 - i] = std::max(indexA, indexB);
        }
    }

    JST_TRACE("[MODULE_ADD] Output shape {}.", outputShape);

    const DeviceType device = tensorA.device();
    const DataType dtype = tensorA.dtype();

    a = tensorA;
    b = tensorB;

    JST_CHECK(a.broadcastTo(outputShape));
    JST_CHECK(b.broadcastTo(outputShape));

    JST_CHECK(c.create(device, dtype, outputShape));

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
