#ifndef JETSTREAM_COMPUTE_GRAPH_METAL_HH
#define JETSTREAM_COMPUTE_GRAPH_METAL_HH

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/generic.hh"

namespace Jetstream {

class Metal : public Graph {
 public:
    Metal();
    ~Metal();

    constexpr Device device() const {
        return Device::Metal;
    }

    Result create();
    Result compute();
    Result computeReady();
    Result destroy();

    constexpr MTL::CommandQueue* commandQueue() const {
        return _commandQueue;
    }

    constexpr MTL::CommandBuffer* commandBuffer() const {
        return _commandBuffer;
    }

    static Result CompileKernel(const char* shaderSrc, 
                                const char* methodName,
                                MTL::ComputePipelineState** pipelineState);

    template<typename ConstantsType>
    static ConstantsType* Constants(auto& assets) {
        return reinterpret_cast<ConstantsType*>(assets.constants.cpu().data());
    }

    template<typename ConstantsType>
    static ConstantsType* CreateConstants(auto& assets) {
        assets.constants = Tensor<Device::Metal, U8>({sizeof(ConstantsType)});
        return Constants<ConstantsType>(assets);
    }

 private:
    NS::AutoreleasePool* innerPool;
    NS::AutoreleasePool* outerPool;

    MTL::CommandQueue* _commandQueue;
    MTL::CommandBuffer* _commandBuffer;
};

}  // namespace Jetstream

#endif
