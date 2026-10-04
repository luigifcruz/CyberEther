#include "lineplot.hh"

#include "resources/shaders/global_shaders.hh"
#include "resources/shaders/signal_view_shaders.hh"

namespace Jetstream::Modules {

void SignalViewLineplot::configure(const Config& nextConfig) {
    config = nextConfig;
    displayedPoints.clear();
    uniformsDirty = false;
}

Result SignalViewLineplot::create(const std::shared_ptr<Render::Window>& window,
                                  const Tensors& tensors) {
    {
        Render::Buffer::Config cfg;
        cfg.buffer = &signalUniforms;
        cfg.elementByteSize = sizeof(signalUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(signalUniformBuffer, cfg));
    }

    auto buildTrace = [&](
        const std::shared_ptr<Render::Buffer>& uniformBuffer,
        Tensor& pointsTensor,
        std::shared_ptr<Render::Buffer>& pointsBuffer,
        Tensor& verticesTensor,
        std::shared_ptr<Render::Buffer>& verticesBuffer,
        std::shared_ptr<Render::Kernel>& kernel,
        std::shared_ptr<Render::Vertex>& vertex,
        std::shared_ptr<Render::Draw>& draw,
        std::shared_ptr<Render::Program>& program) -> Result {
        {
            Render::Buffer::Config cfg;
            cfg.buffer = pointsTensor.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = pointsTensor.size();
            cfg.target = Render::Buffer::Target::STORAGE;
            cfg.enableZeroCopy = false;
            JST_CHECK(window->build(pointsBuffer, cfg));
        }
        {
            Render::Buffer::Config cfg;
            cfg.buffer = verticesTensor.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = verticesTensor.size();
            cfg.target = Render::Buffer::Target::VERTEX |
                         Render::Buffer::Target::STORAGE;
            cfg.enableZeroCopy = false;
            JST_CHECK(window->build(verticesBuffer, cfg));
        }
        {
            Render::Kernel::Config cfg;
            cfg.gridSize = {config.numberOfElements - 1, 1, 1};
            cfg.kernels = GlobalKernelsPackage["thicklinestrip"];
            cfg.buffers = {
                {uniformBuffer, Render::Kernel::AccessMode::READ},
                {pointsBuffer, Render::Kernel::AccessMode::READ},
                {verticesBuffer, Render::Kernel::AccessMode::WRITE},
            };
            JST_CHECK(window->build(kernel, cfg));
        }
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {verticesBuffer, 4},
            };
            JST_CHECK(window->build(vertex, cfg));
        }
        {
            Render::Draw::Config cfg;
            cfg.buffer = vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLE_STRIP;
            JST_CHECK(window->build(draw, cfg));
        }
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["signal"];
            cfg.draws = {draw};
            cfg.buffers = {
                {uniformBuffer,
                 Render::Program::Target::VERTEX |
                     Render::Program::Target::FRAGMENT},
            };
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(program, cfg));
        }
        return Result::SUCCESS;
    };

    JST_CHECK(buildTrace(signalUniformBuffer,
                         tensors.signalPoints, signalPointsBuffer,
                         tensors.signalVertices, signalVerticesBuffer,
                         signalKernel, signalVertex,
                         drawSignalVertex, signalProgram));

    // Fill element (analyser-style persistence area beneath the trace).

    if (config.fill) {
        {
            Render::Buffer::Config cfg;
            cfg.buffer = tensors.fillVertices.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = tensors.fillVertices.size();
            cfg.target = Render::Buffer::Target::VERTEX |
                         Render::Buffer::Target::STORAGE;
            cfg.enableZeroCopy = false;
            JST_CHECK(window->build(fillVerticesBuffer, cfg));
        }

        {
            Render::Kernel::Config cfg;
            cfg.gridSize = {config.numberOfElements, 1, 1};
            cfg.kernels = GlobalKernelsPackage["fillarea"];
            cfg.buffers = {
                {signalUniformBuffer, Render::Kernel::AccessMode::READ},
                {signalPointsBuffer, Render::Kernel::AccessMode::READ},
                {fillVerticesBuffer, Render::Kernel::AccessMode::WRITE},
            };
            JST_CHECK(window->build(fillKernel, cfg));
        }

        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {fillVerticesBuffer, 2},
            };
            JST_CHECK(window->build(fillVertex, cfg));
        }

        {
            Render::Draw::Config cfg;
            cfg.buffer = fillVertex;
            cfg.mode = Render::Draw::Mode::TRIANGLE_STRIP;
            JST_CHECK(window->build(drawFillVertex, cfg));
        }

        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["fill"];
            cfg.draws = {drawFillVertex};
            cfg.buffers = {
                {signalUniformBuffer,
                 Render::Program::Target::VERTEX |
                     Render::Program::Target::FRAGMENT},
            };
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(fillProgram, cfg));
        }
    }

    // Max hold trace (dimmed grey line behind the live trace).

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &holdUniforms;
        cfg.elementByteSize = sizeof(holdUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(holdUniformBuffer, cfg));
    }

    JST_CHECK(buildTrace(holdUniformBuffer,
                         tensors.maxHoldPoints, maxHoldPointsBuffer,
                         tensors.maxHoldVertices, maxHoldVerticesBuffer,
                         maxHoldKernel, maxHoldVertex,
                         drawMaxHoldVertex, maxHoldProgram));

    signalUniforms.traceColor[0] = 1.0f;
    signalUniforms.traceColor[1] = 0.85f;
    signalUniforms.traceColor[2] = 0.0f;
    signalUniforms.traceColor[3] = 0.25f;

    holdUniforms.traceColor[0] = 0.6f;
    holdUniforms.traceColor[1] = 0.45f;
    holdUniforms.traceColor[2] = 0.0f;
    holdUniforms.traceColor[3] = 0.7f;

    return Result::SUCCESS;
}

void SignalViewLineplot::attach(Render::Surface::Config& surface) const {
    surface.kernels.push_back(signalKernel);
    if (config.fill) {
        surface.kernels.push_back(fillKernel);
    }
    if (config.maxHold) {
        surface.kernels.push_back(maxHoldKernel);
    }
    if (config.fill) {
        surface.programs.push_back(fillProgram);
    }
    if (config.maxHold) {
        surface.programs.push_back(maxHoldProgram);
    }
    surface.programs.push_back(signalProgram);
}

void SignalViewLineplot::layout(const glm::mat4& transform,
                                const Extent2D<F32>& pixelSize,
                                const F32 panelScale,
                                const F32 zoom,
                                const Render::ScissorRect& scissor) {
    for (auto* uniforms : {&signalUniforms, &holdUniforms}) {
        uniforms->transform = transform;
        uniforms->thickness[0] = pixelSize.x * detail::kSignalViewLineThickness * 3.0f;
        uniforms->thickness[1] =
            pixelSize.y * detail::kSignalViewLineThickness * 3.0f / panelScale;
        uniforms->zoom = zoom;
        uniforms->numberOfPoints = config.numberOfElements;
    }

    signalProgram->scissorRect(scissor);
    if (config.fill) {
        fillProgram->scissorRect(scissor);
    }
    if (config.maxHold) {
        maxHoldProgram->scissorRect(scissor);
    }

    uniformsDirty = true;
}

Result SignalViewLineplot::upload(const Tensor& signalPoints, bool& holdPending) {
    JST_CHECK(signalPointsBuffer->update());
    if (detail::HostReadable(signalPoints)) {
        const F32* points = signalPoints.data<F32>();
        displayedPoints.assign(points, points + signalPoints.size());
    }
    signalKernel->update();
    if (config.fill) {
        fillKernel->update();
    }
    if (config.maxHold && holdPending) {
        JST_CHECK(maxHoldPointsBuffer->update());
        maxHoldKernel->update();
        holdPending = false;
    }
    return Result::SUCCESS;
}

Result SignalViewLineplot::present() {
    if (!uniformsDirty) {
        return Result::SUCCESS;
    }
    JST_CHECK(signalUniformBuffer->update());
    signalKernel->update();
    if (config.fill) {
        fillKernel->update();
    }
    if (config.maxHold) {
        JST_CHECK(holdUniformBuffer->update());
        maxHoldKernel->update();
    }
    uniformsDirty = false;
    return Result::SUCCESS;
}

std::optional<F32> SignalViewLineplot::sample(const F32 position) const {
    const U64 count = config.numberOfElements;
    if (count < 2 || displayedPoints.size() < count * 2) {
        return std::nullopt;
    }
    const F32 index = (position + 1.0f) * 0.5f * (count - 1);
    const U64 lower = std::min(static_cast<U64>(index), count - 2);
    const F32 fraction = std::clamp(index - static_cast<F32>(lower), 0.0f, 1.0f);
    const F32 yLower = displayedPoints[(lower * 2) + 1];
    const F32 yUpper = displayedPoints[(lower * 2) + 3];
    return yLower + (yUpper - yLower) * fraction;
}

}  // namespace Jetstream::Modules
