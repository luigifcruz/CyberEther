#include "instance_remote_encoder.hh"

#include "jetstream/logger.hh"

#include <string>

#include <drm_fourcc.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/hwcontext_vaapi.h>
}

namespace Jetstream {

struct Instance::Remote::Encoder::VaapiInput : public Input {
    ~VaapiInput() override {
        av_buffer_unref(&frames);
        av_buffer_unref(&device);
    }

    Result create(Encoder& encoder) override {
        const std::string node = "/dev/dri/renderD" + std::to_string(encoder.config.adapter.render);
        if (const int error = av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_DRM, node.c_str(), nullptr, 0);
            error < 0) {
            JST_ERROR("[REMOTE] Failed to open the render node '{}' for VAAPI.", node);
            return Result::ERROR;
        }

        frames = av_hwframe_ctx_alloc(device);
        if (!frames) {
            JST_ERROR("[REMOTE] Failed to allocate the DRM encoder frames.");
            return Result::ERROR;
        }

        auto* framesContext = reinterpret_cast<AVHWFramesContext*>(frames->data);
        framesContext->format = AV_PIX_FMT_DRM_PRIME;
        framesContext->sw_format = AV_PIX_FMT_BGRA;
        framesContext->width = static_cast<int>(encoder.config.size.x);
        framesContext->height = static_cast<int>(encoder.config.size.y);

        if (av_hwframe_ctx_init(frames) < 0) {
            JST_ERROR("[REMOTE] Failed to initialize the DRM encoder frames.");
            return Result::ERROR;
        }

        return encoder.createFilter(frames, "hwmap=derive_device=vaapi,"
                                            "scale_vaapi=format=nv12:out_color_matrix=bt709:out_range=tv"
                                            ":extra_hw_frames=4");
    }

    Result wrap(Encoder& encoder, const SharedFrame& frame, AVFrame* output) override {
        descriptor = {};
        descriptor.nb_objects = 1;
        descriptor.objects[0].fd = static_cast<int>(frame.handle);
        descriptor.objects[0].size = frame.size;
        descriptor.objects[0].format_modifier = DRM_FORMAT_MOD_LINEAR;
        descriptor.nb_layers = 1;
        descriptor.layers[0].format = DRM_FORMAT_ARGB8888;
        descriptor.layers[0].nb_planes = 1;
        descriptor.layers[0].planes[0].object_index = 0;
        descriptor.layers[0].planes[0].offset = 0;
        descriptor.layers[0].planes[0].pitch = static_cast<ptrdiff_t>(frame.pitch);

        output->format = AV_PIX_FMT_DRM_PRIME;
        output->width = encoder.context->width;
        output->height = encoder.context->height;
        output->data[0] = reinterpret_cast<U8*>(&descriptor);
        output->hw_frames_ctx = av_buffer_ref(frames);
        output->buf[0] = encoder.borrow(output->data[0], sizeof(descriptor));
        if (!output->hw_frames_ctx || !output->buf[0]) {
            JST_ERROR("[REMOTE] Failed to wrap the DRM encoder frame.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    bool borrowed() const override {
        return true;
    }

    Result synchronize(AVFrame* frame) override {
        const auto* framesContext = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
        const auto* deviceContext = static_cast<const AVVAAPIDeviceContext*>(framesContext->device_ctx->hwctx);
        const auto surface = static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>(frame->data[3]));

        if (const VAStatus status = vaSyncSurface(deviceContext->display, surface); status != VA_STATUS_SUCCESS) {
            JST_ERROR("[REMOTE] Failed to wait for the VAAPI color conversion ({}).", static_cast<int>(status));
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

 private:
    AVBufferRef* device = nullptr;
    AVBufferRef* frames = nullptr;
    AVDRMFrameDescriptor descriptor = {};
};

std::unique_ptr<Instance::Remote::Encoder::Input> Instance::Remote::Encoder::CreateVaapiInput() {
    return std::make_unique<VaapiInput>();
}

}  // namespace Jetstream
