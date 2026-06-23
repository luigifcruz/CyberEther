#include "jetstream/parser.hh"

#include "jetstream/memory/types.hh"
#include "jetstream/runtime.hh"
#include "jetstream/scheduler.hh"

namespace Jetstream {

//
// TypedToString
//

Result Parser::TypedToString(const std::any& variable, std::string& encoded) {
    if (variable.type() == typeid(std::string)) {
        const auto& stringValue = std::any_cast<std::string>(variable);
        encoded = jst::fmt::format("{}", stringValue);
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(I32)) {
        encoded = jst::fmt::format("{}", std::any_cast<I32>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(U64)) {
        encoded = jst::fmt::format("{}", std::any_cast<U64>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(F32)) {
        encoded = jst::fmt::format("{}", std::any_cast<F32>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(F64)) {
        encoded = jst::fmt::format("{}", std::any_cast<F64>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(CF32)) {
        const auto& complex = std::any_cast<CF32>(variable);
        encoded = jst::fmt::format("{}{}{}",
                                   complex.real(),
                                   complex.imag() < 0 ? "-" : "+",
                                   std::abs(complex.imag()));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(bool)) {
        encoded = jst::fmt::format("{}", std::any_cast<bool>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(DeviceType)) {
        encoded = GetDeviceName(std::any_cast<DeviceType>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(RuntimeType)) {
        encoded = GetRuntimeName(std::any_cast<RuntimeType>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(SchedulerType)) {
        encoded = GetSchedulerName(std::any_cast<SchedulerType>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(std::vector<U64>)) {
        const auto& values = std::any_cast<std::vector<U64>>(variable);
        encoded = jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(std::vector<F32>)) {
        const auto& values = std::any_cast<std::vector<F32>>(variable);
        encoded = jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(std::vector<F64>)) {
        const auto& values = std::any_cast<std::vector<F64>>(variable);
        encoded = jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(Range<F32>)) {
        const auto& range = std::any_cast<Range<F32>>(variable);
        encoded = jst::fmt::format("[{}, {}]", range.min, range.max);
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(Extent2D<U64>)) {
        const auto& size = std::any_cast<Extent2D<U64>>(variable);
        encoded = jst::fmt::format("[{}, {}]", size.x, size.y);
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(Extent2D<F32>)) {
        const auto& size = std::any_cast<Extent2D<F32>>(variable);
        encoded = jst::fmt::format("[{}, {}]", size.x, size.y);
        return Result::SUCCESS;
    }

    JST_ERROR("[PARSER] Failed to serialize variable. Check if the input and output are compatible.");
    JST_TRACE("[PARSER] Variable type: {}", variable.type().name());
    return Result::ERROR;
}

}  // namespace Jetstream
