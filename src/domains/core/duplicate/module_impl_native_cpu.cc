#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct DuplicateImplNativeCpu : public DuplicateImpl,
                                public NativeCpuRuntimeContext,
                                public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

  private:
    template<typename T>
    Result kernelCopy();

    std::function<Result()> kernel;
};

Result DuplicateImplNativeCpu::create() {
    JST_CHECK(DuplicateImpl::create());

    switch (input.dtype()) {
        case DataType::F32:  kernel = [this]() { return kernelCopy<F32>();  }; break;
        case DataType::F64:  kernel = [this]() { return kernelCopy<F64>();  }; break;
        case DataType::I8:   kernel = [this]() { return kernelCopy<I8>();   }; break;
        case DataType::I16:  kernel = [this]() { return kernelCopy<I16>();  }; break;
        case DataType::I32:  kernel = [this]() { return kernelCopy<I32>();  }; break;
        case DataType::I64:  kernel = [this]() { return kernelCopy<I64>();  }; break;
        case DataType::U8:   kernel = [this]() { return kernelCopy<U8>();   }; break;
        case DataType::U16:  kernel = [this]() { return kernelCopy<U16>();  }; break;
        case DataType::U32:  kernel = [this]() { return kernelCopy<U32>();  }; break;
        case DataType::U64:  kernel = [this]() { return kernelCopy<U64>();  }; break;
        case DataType::CF32: kernel = [this]() { return kernelCopy<CF32>(); }; break;
        case DataType::CF64: kernel = [this]() { return kernelCopy<CF64>(); }; break;
        case DataType::CI8:  kernel = [this]() { return kernelCopy<CI8>();  }; break;
        case DataType::CI16: kernel = [this]() { return kernelCopy<CI16>(); }; break;
        case DataType::CI32: kernel = [this]() { return kernelCopy<CI32>(); }; break;
        case DataType::CI64: kernel = [this]() { return kernelCopy<CI64>(); }; break;
        case DataType::CU8:  kernel = [this]() { return kernelCopy<CU8>();  }; break;
        case DataType::CU16: kernel = [this]() { return kernelCopy<CU16>(); }; break;
        case DataType::CU32: kernel = [this]() { return kernelCopy<CU32>(); }; break;
        case DataType::CU64: kernel = [this]() { return kernelCopy<CU64>(); }; break;
        case DataType::None:
            break;
    }

    if (!kernel) {
        JST_ERROR("[MODULE_DUPLICATE_NATIVE_CPU] Unsupported data type '{}'.", input.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DuplicateImplNativeCpu::computeSubmit() {
    return kernel();
}

template<typename T>
Result DuplicateImplNativeCpu::kernelCopy() {
    if (input.contiguous() && input.sizeBytes() == input.buffer().sizeBytes()) {
        JST_CHECK(output.copyFrom(input));
    } else {
        JST_CHECK(AutomaticIterator<const T, T>(
            [](const auto& in, auto& out) {
                out = in;
            },
        input, output));
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(DuplicateImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
