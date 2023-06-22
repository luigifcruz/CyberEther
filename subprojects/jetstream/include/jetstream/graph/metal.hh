#ifndef JETSTREAM_GRAPH_METAL_HH
#define JETSTREAM_GRAPH_METAL_HH

#include "jetstream/memory/base.hh"
#include "jetstream/graph/generic.hh"

namespace Jetstream {

class Metal : public Graph {
 public:
    Metal();

    constexpr Device device() const {
        return Device::Metal;
    }

    Result createCompute();
    Result compute();
    Result destroyCompute();

    static Result CompileKernel(const char* shaderSrc, 
                                const char* methodName,
                                MTL::ComputePipelineState** pipelineState);

    template<typename ConstantsType>
    static ConstantsType* Constants(auto& assets) {
        return reinterpret_cast<ConstantsType*>(assets.constants.data());
    }

    template<typename ConstantsType>
    static ConstantsType* CreateConstants(auto& assets) {
        assets.constants = Vector<Device::Metal, U8>({sizeof(ConstantsType)});
        return Constants<ConstantsType>(assets);
    }

 private:
    NS::AutoreleasePool* innerPool;
    NS::AutoreleasePool* outerPool;
};

}  // namespace Jetstream

#endif
