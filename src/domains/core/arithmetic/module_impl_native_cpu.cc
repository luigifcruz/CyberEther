#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ArithmeticImplNativeCpu : public ArithmeticImpl,
                                public Runtime::Context,
                                public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelAddF32();
    Result kernelSubF32();
    Result kernelMulF32();
    Result kernelDivF32();

    Result kernelAddCF32();
    Result kernelSubCF32();
    Result kernelMulCF32();
    Result kernelDivCF32();

    std::function<Result()> zeroKernel;
    std::function<Result()> kernel;
};

Result ArithmeticImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(ArithmeticImpl::create());

    // Register compute kernel.

    if (input.dtype() == DataType::F32) {
        zeroKernel = [this]() {
            return AutomaticIterator<F32>(
                [](auto& out) { out = 0.0f; },
            broadcastedOutput);
        };

        if (operation == "add") {
            kernel = [this]() { return kernelAddF32(); };
        } else if (operation == "sub") {
            kernel = [this]() { return kernelSubF32(); };
        } else if (operation == "mul") {
            kernel = [this]() { return kernelMulF32(); };
        } else if (operation == "div") {
            kernel = [this]() { return kernelDivF32(); };
        }

        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::CF32) {
        zeroKernel = [this]() {
            return AutomaticIterator<CF32>(
                [](auto& out) { out = CF32(0.0f, 0.0f); },
            broadcastedOutput);
        };

        if (operation == "add") {
            kernel = [this]() { return kernelAddCF32(); };
        } else if (operation == "sub") {
            kernel = [this]() { return kernelSubCF32(); };
        } else if (operation == "mul") {
            kernel = [this]() { return kernelMulCF32(); };
        } else if (operation == "div") {
            kernel = [this]() { return kernelDivCF32(); };
        }

        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_ARITHMETIC_NATIVE_CPU] Unsupported data type '{}'.",
              input.dtype());
    return Result::ERROR;
}

Result ArithmeticImplNativeCpu::computeSubmit() {
    JST_CHECK(zeroKernel());
    return kernel();
}

Result ArithmeticImplNativeCpu::kernelAddF32() {
    return AutomaticIterator<F32, F32>(
        [](auto& out, const auto& in) {
            out += in;
        },
    broadcastedOutput, input);
}

Result ArithmeticImplNativeCpu::kernelSubF32() {
    return AutomaticIterator<F32, F32>(
        [](auto& out, const auto& in) {
            out -= in;
        },
    broadcastedOutput, input);
}

Result ArithmeticImplNativeCpu::kernelMulF32() {
    return AutomaticIterator<F32, F32>(
        [](auto& out, const auto& in) {
            out *= in;
        },
    broadcastedOutput, input);
}

Result ArithmeticImplNativeCpu::kernelDivF32() {
    return AutomaticIterator<F32, F32>(
        [](auto& out, const auto& in) {
            out /= in;
        },
    broadcastedOutput, input);
}

Result ArithmeticImplNativeCpu::kernelAddCF32() {
    return AutomaticIterator<CF32, CF32>(
        [](auto& out, const auto& in) {
            out += in;
        },
    broadcastedOutput, input);
}

Result ArithmeticImplNativeCpu::kernelSubCF32() {
    return AutomaticIterator<CF32, CF32>(
        [](auto& out, const auto& in) {
            out -= in;
        },
    broadcastedOutput, input);
}

Result ArithmeticImplNativeCpu::kernelMulCF32() {
    return AutomaticIterator<CF32, CF32>(
        [](auto& out, const auto& in) {
            out *= in;
        },
    broadcastedOutput, input);
}

Result ArithmeticImplNativeCpu::kernelDivCF32() {
    return AutomaticIterator<CF32, CF32>(
        [](auto& out, const auto& in) {
            out /= in;
        },
    broadcastedOutput, input);
}

JST_REGISTER_MODULE(ArithmeticImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
