#include "module_impl.hh"

namespace Jetstream::Modules {

Result FrbnnDetectImpl::define() {
    JST_CHECK(defineInterfaceInput("probabilities"));
    JST_CHECK(defineInterfaceOutput("signal"));
    return Result::SUCCESS;
}

Result FrbnnDetectImpl::create() {
    input = inputs().at("probabilities").tensor;

    const auto& shape = input.shape();
    if (shape.size() == 1) {
        inputIs2D = false;
    } else if (shape.size() == 2) {
        inputIs2D = true;
        if (classIndex >= shape[1]) {
            JST_ERROR("[MODULE_FRBNN_DETECT] classIndex={} is out of range for {} output classes.",
                      classIndex, shape[1]);
            return Result::ERROR;
        }
    } else {
        JST_ERROR("[MODULE_FRBNN_DETECT] Expected 1-D or 2-D probability tensor, got {}D.", shape.size());
        return Result::ERROR;
    }

    const U64 batchDim = shape[0];
    JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {batchDim}));
    outputs()["signal"].produced(name(), "signal", output);

    snapshotTotalCandidates.publish(0);
    snapshotLatestProbability.publish(0.0f);

    return Result::SUCCESS;
}

Result FrbnnDetectImpl::reconfigure() {
    const auto& cfg = *candidate();

    if (cfg.classIndex != classIndex) {
        return Result::RECREATE;
    }

    threshold = cfg.threshold;
    return Result::SUCCESS;
}

U64 FrbnnDetectImpl::getTotalCandidates() const {
    return snapshotTotalCandidates.get();
}

F32 FrbnnDetectImpl::getLatestProbability() const {
    return snapshotLatestProbability.get();
}

}  // namespace Jetstream::Modules
