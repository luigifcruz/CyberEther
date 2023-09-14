#include "jetstream/render/webgpu/vertex.hh"
#include "jetstream/render/webgpu/draw.hh"

namespace Jetstream::Render {

using Implementation = DrawImp<Device::WebGPU>;

Implementation::DrawImp(const Config& config) : Draw(config) {
    buffer = std::dynamic_pointer_cast<VertexImp<Device::WebGPU>>(config.buffer);
}

Result Implementation::create(wgpu::RenderPipelineDescriptor& renderDescriptor) {
    JST_DEBUG("[WebGPU] Creating draw.");

    auto topology = wgpu::PrimitiveTopology::PointList;
        
    switch (config.mode) {
        case Mode::TRIANGLE_FAN:
            topology = wgpu::PrimitiveTopology::TriangleStrip;
            break;
        case Mode::TRIANGLES:
            topology = wgpu::PrimitiveTopology::TriangleList;
            break;
        case Mode::LINES:
            topology = wgpu::PrimitiveTopology::LineList;
            break;
        case Mode::LINE_STRIP:
            topology = wgpu::PrimitiveTopology::LineStrip;
            break;
        case Mode::POINTS:
            topology = wgpu::PrimitiveTopology::PointList;
            break;
    }

    renderDescriptor.primitive.frontFace = wgpu::FrontFace::CCW;
	  renderDescriptor.primitive.cullMode = wgpu::CullMode::None;
	  renderDescriptor.primitive.topology = topology;
	  renderDescriptor.primitive.stripIndexFormat = wgpu::IndexFormat::Undefined;

    JST_CHECK(buffer->create(renderDescriptor));

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying draw.");

    JST_CHECK(buffer->destroy());

    return Result::SUCCESS;
}

Result Implementation::encode(wgpu::RenderPassEncoder& renderPassEncoder) {
    JST_CHECK(buffer->encode(renderPassEncoder));

    if (buffer->isBuffered()) {
        renderPassEncoder.DrawIndexed(buffer->getVertexCount());
    } else {
        renderPassEncoder.Draw(buffer->getVertexCount());
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
