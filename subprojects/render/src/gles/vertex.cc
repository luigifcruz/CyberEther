#include "render/gles/vertex.hpp"

namespace Render {

Result GLES::Vertex::create() {
    glGenVertexArrays(1, &vao);

    this->start();
    int i = 0;
    bool cudaEnabled = true;
    for (auto& buffer : cfg.buffers) {
        uint usage = GL_STATIC_DRAW;
        switch (buffer.usage) {
            case Vertex::Buffer::Usage::Dynamic:
                usage = GL_DYNAMIC_DRAW;
                break;
            case Vertex::Buffer::Usage::Stream:
                usage = GL_STREAM_DRAW;
                break;
            case Vertex::Buffer::Usage::Static:
                usage = GL_STATIC_DRAW;
                break;
        }

        glGenBuffers(1, &buffer.index);
        glBindBuffer(GL_ARRAY_BUFFER, buffer.index);
        auto ptr = (buffer.cudaInterop) ? nullptr : buffer.data;
        glBufferData(GL_ARRAY_BUFFER, buffer.size * sizeof(float), ptr, usage);
        glVertexAttribPointer(i, buffer.stride, GL_FLOAT, GL_FALSE, buffer.stride * sizeof(float), 0);
        glEnableVertexAttribArray(i++);
        vertex_count = buffer.size / buffer.stride;

        if (buffer.cudaInterop) {
#ifdef RENDER_CUDA_AVAILABLE
            CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&buffer._cuda_res, buffer.index,
                    cudaGraphicsMapFlagsWriteDiscard));
            cudaEnabled = true;
#endif
        }
    }

    if (cudaEnabled) {
#ifdef RENDER_CUDA_AVAILABLE
        int leastPriority = -1, greatestPriority = -1;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
#endif
    }

    if (cfg.indices.size() != 0) {
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, cfg.indices.size() * sizeof(uint),
                cfg.indices.data(), GL_STATIC_DRAW);
        vertex_count = cfg.indices.size();
    }
    this->end();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::destroy() {
    bool cudaEnabled = false;

    for (auto& buffer : cfg.buffers) {
        glDeleteBuffers(1, &buffer.index);
#ifdef RENDER_CUDA_AVAILABLE
        if (buffer._cuda_res == nullptr) {
            cudaGraphicsUnregisterResource(buffer._cuda_res);
            cudaEnabled = true;
        }
#endif
    }

    if (cudaEnabled) {
#ifdef RENDER_CUDA_AVAILABLE
        cudaStreamDestroy(stream);
#endif
    }

    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::start() {
    glBindVertexArray(vao);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::end() {
    glBindVertexArray(0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::update() {
    this->start();
    for (auto& buffer : cfg.buffers) {
        if (buffer.cudaInterop) {
#ifdef RENDER_CUDA_AVAILABLE
            float *buffer_ptr;
            size_t buffer_len;
            CUDA_CHECK(cudaGraphicsMapResources(1, &buffer._cuda_res, stream));
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&buffer_ptr, &buffer_len, buffer._cuda_res));
            CUDA_CHECK(cudaMemcpyAsync(buffer_ptr, buffer.data, buffer_len, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &buffer._cuda_res, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
            break;
        }

        glBindBuffer(GL_ARRAY_BUFFER, buffer.index);
        glBufferSubData(GL_ARRAY_BUFFER, 0, buffer.size * sizeof(float), buffer.data);
    }
    this->end();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

uint GLES::Vertex::buffered() {
    return cfg.indices.size() != 0;
}

uint GLES::Vertex::count() {
    return vertex_count;
}

} // namespace Render
