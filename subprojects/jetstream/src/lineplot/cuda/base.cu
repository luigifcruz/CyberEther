#include "jetstream/lineplot/cuda.hpp"

namespace Jetstream::Lineplot {

float map(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}


CUDA::CUDA(Config& c) : Generic(c) {
    for (float i = -1.0f; i < +1.0f; i += 0.10f) {
        grid.push_back(-1.0f);
        grid.push_back(i);
        grid.push_back(+0.0f);
        grid.push_back(+1.0f);
        grid.push_back(i);
        grid.push_back(+0.0f);
        grid.push_back(i);
        grid.push_back(-1.0f);
        grid.push_back(+0.0f);
        grid.push_back(i);
        grid.push_back(+1.0f);
        grid.push_back(+0.0f);
    }

    auto render = cfg.render;

    out_len = in.buf.size() * 3 ;
    cudaMallocManaged(&out_dptr, out_len * sizeof(float));

    for (int i = 0; i < out_len; i += 3) {
        out_dptr[(int)i*1] = (float)map((float)i / out_len, 0.0, 1.0, -1.0, +1.0);
        out_dptr[i*2] = 0.0;
        out_dptr[i*3] = 0.0;
    }

    std::cout << out_dptr[0] << std::endl;
    out_dptr[0] = -1.0;
    std::cout << out_dptr[0] << std::endl;

    Render::Vertex::Buffer gridVbo;
    gridVbo.data = grid.data();
    gridVbo.size = grid.size();
    gridVbo.stride = 3;
    gridVbo.usage = Render::Vertex::Buffer::Static;
    gridVertexCfg.buffers = {gridVbo};
    gridVertex = render->create(gridVertexCfg);

    drawGridVertexCfg.buffer = gridVertex;
    drawGridVertexCfg.mode = Render::Draw::Lines;
    drawGridVertex = render->create(drawGridVertexCfg);

    Render::Vertex::Buffer plotVbo;
    plotVbo.data = out_dptr;
    plotVbo.size = out_len;
    plotVbo.stride = 3;
    plotVbo.usage = Render::Vertex::Buffer::Dynamic;
    plotVbo.cudaInterop = true;
    lineVertexCfg.buffers = {plotVbo};
    lineVertex = render->create(lineVertexCfg);

    drawLineVertexCfg.buffer = lineVertex;
    drawLineVertexCfg.mode = Render::Draw::LineStrip;
    drawLineVertex = render->create(drawLineVertexCfg);

    lutTextureCfg.height = 1;
    lutTextureCfg.width = 256;
    lutTextureCfg.buffer = (uint8_t*)turbo_srgb_bytes;
    lutTextureCfg.key = "LutTexture";
    lutTexture = render->create(lutTextureCfg);

    programCfg.vertexSource = &vertexSource;
    programCfg.fragmentSource = &fragmentSource;
    programCfg.draws = {drawGridVertex, drawLineVertex};
    programCfg.textures = {lutTexture};
    program = render->create(programCfg);

    textureCfg.width = cfg.width;
    textureCfg.height = cfg.height;
    texture = render->create(textureCfg);

    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    surface = render->createAndBind(surfaceCfg);
}

CUDA::~CUDA() {
}

void cudaMemcpyStrided(
        void *dst, int dstStride,
        void *src, int srcStride,
        int numElements, int elementSize, enum cudaMemcpyKind kind) {
    size_t srcPitchInBytes = srcStride * elementSize;
    size_t dstPitchInBytes = dstStride * elementSize;
    size_t width = 1 * elementSize;
    size_t height = numElements;
    cudaMemcpy2D(
        dst, dstPitchInBytes,
        src, srcPitchInBytes,
        width, height,
        kind);
}

Result CUDA::_compute() {
    cudaMemcpyStrided(out_dptr+1, 3, in.buf.data(), 1, in.buf.size(), sizeof(float), cudaMemcpyDeviceToDevice);

    return Result::SUCCESS;
}

Result CUDA::_present() {
    lineVertex->update();

    return Result::SUCCESS;
}

} // namespace Jetstream::Lineplot
