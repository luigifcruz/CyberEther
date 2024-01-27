#ifndef JETSTREAM_COMPUTE_GRAPH_CUDA_HH
#define JETSTREAM_COMPUTE_GRAPH_CUDA_HH

#include "jetstream/compute/graph/generic.hh"

namespace Jetstream {

class CUDA : public Graph {
 public:
    CUDA();
    ~CUDA();

    constexpr Device device() const {
        return Device::CUDA;
    }

    constexpr const cudaStream_t& stream() const {
        return _stream;
    }

    Result create();
    Result compute();
    Result computeReady();
    Result destroy();

    enum class KernelHeader {
        NONE,
        COMPLEX,
    };

    Result createKernel(const std::string& name, 
                        const std::string& source,
                        const std::vector<KernelHeader>& headers = {});

    Result launchKernel(const std::string& name, 
                        const std::vector<U64>& grid,
                        const std::vector<U64>& block,
                        void** arguments);

    Result destroyKernel(const std::string& name);

 private:
    cudaStream_t _stream;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    
};

}  // namespace Jetstream

#endif
