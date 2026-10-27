#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include <jetstream/memory/macros.hh>
#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

template<typename T>
struct TypeTag {
    using type = T;
};

template<typename Function>
Result VisitDataType(const DataType& dtype, Function&& function) {
    switch (dtype) {
        case DataType::F32: return function(TypeTag<F32>{});
        case DataType::F64: return function(TypeTag<F64>{});
        case DataType::I8: return function(TypeTag<I8>{});
        case DataType::I16: return function(TypeTag<I16>{});
        case DataType::I32: return function(TypeTag<I32>{});
        case DataType::I64: return function(TypeTag<I64>{});
        case DataType::U8: return function(TypeTag<U8>{});
        case DataType::U16: return function(TypeTag<U16>{});
        case DataType::U32: return function(TypeTag<U32>{});
        case DataType::U64: return function(TypeTag<U64>{});
        case DataType::CF32: return function(TypeTag<CF32>{});
        case DataType::CF64: return function(TypeTag<CF64>{});
        case DataType::CI8: return function(TypeTag<CI8>{});
        case DataType::CI16: return function(TypeTag<CI16>{});
        case DataType::CI32: return function(TypeTag<CI32>{});
        case DataType::CI64: return function(TypeTag<CI64>{});
        case DataType::CU8: return function(TypeTag<CU8>{});
        case DataType::CU16: return function(TypeTag<CU16>{});
        case DataType::CU32: return function(TypeTag<CU32>{});
        case DataType::CU64: return function(TypeTag<CU64>{});
        default: return Result::ERROR;
    }
}

template<typename T>
constexpr bool IsComplex = !std::is_same_v<T, typename NumericTypeInfo<T>::subtype>;

template<typename T>
constexpr bool NeedsDoubleWork = std::is_same_v<T, F64> ||
                                 (std::is_integral_v<T> && sizeof(T) > 2);

}  // namespace

struct CastImplNativeCpu : public CastImpl,
                           public NativeCpuRuntimeContext,
                           public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;

 private:
    template<typename In, typename Out>
    Result kernelConvert();

    std::function<Result()> kernel;
};

Result CastImplNativeCpu::validate() {
    JST_CHECK(CastImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& candidateInput = inputs().at("buffer").tensor;
    if (!candidateInput.validShape() || candidateInput.size() == 0) {
        return Result::SUCCESS;
    }

    if (validatedBypass) {
        return Result::SUCCESS;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                        alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_CAST_NATIVE_CPU] Output allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result CastImplNativeCpu::create() {
    JST_CHECK(CastImpl::create());

    if (bypass) {
        return Result::SUCCESS;
    }

    return VisitDataType(input.dtype(), [&](auto inputTag) {
        return VisitDataType(outputDtype, [&](auto outputTag) {
            using In = typename decltype(inputTag)::type;
            using Out = typename decltype(outputTag)::type;

            if constexpr (IsComplex<In> && !IsComplex<Out>) {
                JST_ERROR("[MODULE_CAST_NATIVE_CPU] Cannot cast complex '{}' to real '{}'.",
                          input.dtype(), outputDtype);
                return Result::ERROR;
            } else {
                kernel = [this]() { return kernelConvert<In, Out>(); };
                return Result::SUCCESS;
            }
        });
    });
}

Result CastImplNativeCpu::computeSubmit() {
    if (bypass) {
        return Result::SUCCESS;
    }

    return kernel();
}

template<typename In, typename Out>
Result CastImplNativeCpu::kernelConvert() {
    using InScalar = typename NumericTypeInfo<In>::subtype;
    using OutScalar = typename NumericTypeInfo<Out>::subtype;
    using Work = std::conditional_t<NeedsDoubleWork<InScalar> ||
                                    NeedsDoubleWork<OutScalar>, F64, F32>;

    const U64 inputHalf = inputScale.integer ? U64{1} << (inputScale.bits - 1) : 0;
    const U64 outputHalf = outputScale.integer ? U64{1} << (outputScale.bits - 1) : 0;
    const I64 outputMax = outputScale.integer ?
        std::numeric_limits<I64>::max() >> (64 - outputScale.bits) : 0;
    const I64 shift = static_cast<I64>(outputScale.bits) - static_cast<I64>(inputScale.bits);
    const OutScalar inputGain = static_cast<OutScalar>(1.0 / inputScale.fullScale);
    const Work outputGain = static_cast<Work>(outputScale.fullScale);

    const auto decode = [=](const InScalar value) -> I64 {
        if constexpr (std::is_unsigned_v<InScalar>) {
            return static_cast<I64>(static_cast<U64>(value) - inputHalf);
        } else {
            return static_cast<I64>(value);
        }
    };

    const auto encode = [=](const I64 value) -> OutScalar {
        if constexpr (std::is_unsigned_v<OutScalar>) {
            return static_cast<OutScalar>(static_cast<U64>(value) + outputHalf);
        } else {
            return static_cast<OutScalar>(value);
        }
    };

    const auto convert = [=](const InScalar value) -> OutScalar {
        if constexpr (std::is_floating_point_v<InScalar> &&
                      std::is_floating_point_v<OutScalar>) {
            return static_cast<OutScalar>(value);
        } else if constexpr (std::is_floating_point_v<OutScalar>) {
            return static_cast<OutScalar>(decode(value)) * inputGain;
        } else if constexpr (std::is_floating_point_v<InScalar>) {
            Work level = static_cast<Work>(value);
            if (level != level) {
                level = 0;
            }
            const Work scaled = std::nearbyint(level * outputGain);
            if (scaled >= outputGain) {
                return encode(outputMax);
            }
            if (scaled < -outputGain) {
                return encode(-outputMax - 1);
            }
            return encode(static_cast<I64>(scaled));
        } else {
            const I64 level = decode(value);
            if (shift >= 0) {
                return encode(level * (I64{1} << shift));
            }
            const I64 step = I64{1} << -shift;
            I64 quotient = level >> -shift;
            const I64 remainder = level - quotient * step;
            if (remainder > step / 2 || (remainder == step / 2 && (quotient & 1))) {
                quotient += 1;
            }
            return encode(std::min(quotient, outputMax));
        }
    };

    if constexpr (IsComplex<In>) {
        return AutomaticIterator<const In, Out>(
            [=](const auto& in, auto& out) {
                out = Out(convert(in.real()), convert(in.imag()));
            },
        input, output);
    } else if constexpr (IsComplex<Out>) {
        OutScalar zero = 0;
        if constexpr (std::is_integral_v<OutScalar>) {
            zero = encode(0);
        }
        return AutomaticIterator<const In, Out>(
            [=](const auto& in, auto& out) {
                out = Out(convert(in), zero);
            },
        input, output);
    } else {
        return AutomaticIterator<const In, Out>(
            [=](const auto& in, auto& out) {
                out = convert(in);
            },
        input, output);
    }
}

JST_REGISTER_MODULE(CastImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
