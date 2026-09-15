#include "module_impl.hh"

#include <algorithm>
#include <utility>

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result RemoveIndicesImpl::validate() {
    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const auto& config = *candidate();
    const auto candidateAxis = ResolveAxis(config.axis, inputTensor.rank());
    if (!candidateAxis) {
        JST_ERROR("[MODULE_REMOVE_INDICES] Axis {} out of range for tensor with {} dimensions.",
                  config.axis, inputTensor.rank());
        return Result::ERROR;
    }

    const U64 axisSize = inputTensor.shape(*candidateAxis);
    auto removedIndices = config.indices;
    for (const auto index : removedIndices) {
        if (index >= axisSize) {
            JST_ERROR("[MODULE_REMOVE_INDICES] Index {} out of range for axis dimension {}.",
                      index, axisSize);
            return Result::ERROR;
        }
    }

    std::sort(removedIndices.begin(), removedIndices.end());
    removedIndices.erase(std::unique(removedIndices.begin(), removedIndices.end()),
                         removedIndices.end());
    if (removedIndices.size() == axisSize) {
        JST_ERROR("[MODULE_REMOVE_INDICES] Cannot remove every entry along an axis.");
        return Result::ERROR;
    }

    validatedResolvedAxis = *candidateAxis;
    validatedRemovedIndices = std::move(removedIndices);
    return Result::SUCCESS;
}

Result RemoveIndicesImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));
    return Result::SUCCESS;
}

Result RemoveIndicesImpl::create() {
    input = inputs().at("buffer").tensor;
    resolvedAxis = validatedResolvedAxis;

    Shape outputShape = input.shape();
    outputShape[resolvedAxis] -= validatedRemovedIndices.size();

    keptIndices.clear();
    keptIndices.reserve(outputShape[resolvedAxis]);
    auto removed = validatedRemovedIndices.begin();
    for (U64 index = 0; index < input.shape(resolvedAxis); ++index) {
        if (removed != validatedRemovedIndices.end() && *removed == index) {
            ++removed;
        } else {
            keptIndices.push_back(index);
        }
    }

    JST_CHECK(output.create(device(), input.dtype(), outputShape));
    JST_CHECK(output.propagateAttributes(input));
    outputs()["buffer"].produced(name(), "buffer", output);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
