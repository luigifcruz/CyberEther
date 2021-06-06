#include "jetstream/fft/cuda.hpp"

namespace Jetstream::FFT {

CUDA::CUDA(Config& c) : Generic(c) {
  size_t men_len = in.buf.size() * sizeof(float) * 2;
  cudaMallocManaged(&fft_dptr, men_len);
  cufftPlan1d(&plan, in.buf.size(), CUFFT_C2C, 1);

  fft = tcb::span{(std::complex<float>*)fft_dptr, in.buf.size()};
  buf_out.resize(in.buf.size());
  out.buf = buf_out;
}

CUDA::~CUDA() {
  cufftDestroy(plan);
  cudaFree(fft_dptr);
}

Result CUDA::underlyingCompute() {
  std::copy(in.buf.begin(), in.buf.end(), fft.begin());
  cudaDeviceSynchronize();

  cufftExecC2C(plan, fft_dptr, fft_dptr, CUFFT_FORWARD);
  cudaDeviceSynchronize();

  for (int i = 0; i < fft.size(); i++) {
      int ix;
      int stride = fft.size() / 2;

      if (i < stride) {
          ix = stride + i;
      } else {
          ix = i - stride;
      }

      buf_out[i] = 20 * log10(abs(fft[ix]) / fft.size());
      buf_out[i] = (buf_out[i] - cfg.min_db)/(cfg.max_db - cfg.min_db);

      if (buf_out[i] < 0.0) {
          buf_out[i] = 0.0;
      }

      if (buf_out[i] > 1.0) {
          buf_out[i] = 1.0;
      }
  }

  return Result::SUCCESS;
}

Result CUDA::underlyingPresent() {
  return Result::SUCCESS;
}

} // namespace Jetstream::FFT
