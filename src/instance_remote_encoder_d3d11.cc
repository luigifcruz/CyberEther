#include "instance_remote_encoder.hh"

#include "jetstream/logger.hh"

#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>

#include <d3d11_1.h>
#include <dxgi1_2.h>
#ifdef ERROR
#undef ERROR
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}

namespace Jetstream {

struct Instance::Remote::Encoder::D3D11Input : public Input {
    ~D3D11Input() override {
        for (const auto& [handle, texture] : textures) {
            (void)handle;
            texture->Release();
        }
        textures.clear();
        if (query) {
            query->Release();
        }
        if (device1) {
            device1->Release();
        }
        av_buffer_unref(&frames);
        av_buffer_unref(&device);
    }

    Result create(Encoder& encoder) override {
        int index = -1;
        JST_CHECK(findAdapter(encoder.config.adapter, index));

        const std::string ordinal = std::to_string(index);
        if (av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_D3D11VA, ordinal.c_str(), nullptr, 0) < 0) {
            JST_ERROR("[REMOTE] Failed to create the D3D11 encoder device.");
            return Result::ERROR;
        }

        auto* hwctx = deviceContext();
        if (FAILED(hwctx->device->QueryInterface(__uuidof(ID3D11Device1), reinterpret_cast<void**>(&device1)))) {
            JST_ERROR("[REMOTE] D3D11 encoder device can't open shared textures.");
            return Result::ERROR;
        }

        D3D11_QUERY_DESC queryInfo = {};
        queryInfo.Query = D3D11_QUERY_EVENT;
        if (FAILED(hwctx->device->CreateQuery(&queryInfo, &query))) {
            JST_ERROR("[REMOTE] Failed to create the D3D11 encoder fence.");
            return Result::ERROR;
        }

        frames = av_hwframe_ctx_alloc(device);
        if (!frames) {
            JST_ERROR("[REMOTE] Failed to allocate the D3D11 encoder frames.");
            return Result::ERROR;
        }

        auto* framesContext = reinterpret_cast<AVHWFramesContext*>(frames->data);
        framesContext->format = AV_PIX_FMT_D3D11;
        framesContext->sw_format = AV_PIX_FMT_BGRA;
        framesContext->width = Align(static_cast<int>(encoder.config.size.x));
        framesContext->height = Align(static_cast<int>(encoder.config.size.y));

        auto* framesHwctx = static_cast<AVD3D11VAFramesContext*>(framesContext->hwctx);
        framesHwctx->BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

        if (av_hwframe_ctx_init(frames) < 0) {
            JST_ERROR("[REMOTE] Failed to initialize the D3D11 encoder frames.");
            return Result::ERROR;
        }

        if (encoder.config.encoder == EncoderType::QSV) {
            return encoder.createFilter(frames, "hwmap=derive_device=qsv,"
                                                "vpp_qsv=format=nv12:out_color_matrix=bt709:out_range=tv"
                                                ":async_depth=0:extra_hw_frames=4");
        }

        encoder.context->pix_fmt = AV_PIX_FMT_D3D11;
        encoder.context->hw_frames_ctx = av_buffer_ref(frames);
        if (!encoder.context->hw_frames_ctx) {
            JST_ERROR("[REMOTE] Failed to reference the D3D11 encoder frames.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    Result wrap(Encoder& encoder, const SharedFrame& frame, AVFrame* output) override {
        ID3D11Texture2D* shared = nullptr;
        JST_CHECK(open(frame, shared));

        if (av_hwframe_get_buffer(frames, output, 0) < 0) {
            JST_ERROR("[REMOTE] Failed to get a D3D11 encoder frame.");
            return Result::ERROR;
        }
        output->width = encoder.context->width;
        output->height = encoder.context->height;

        auto* hwctx = deviceContext();
        auto* target = reinterpret_cast<ID3D11Texture2D*>(output->data[0]);
        const auto subresource = static_cast<UINT>(reinterpret_cast<intptr_t>(output->data[1]));

        hwctx->lock(hwctx->lock_ctx);
        hwctx->device_context->CopySubresourceRegion(target, subresource, 0, 0, 0, shared, 0, nullptr);
        hwctx->device_context->End(query);
        hwctx->device_context->Flush();
        hwctx->unlock(hwctx->lock_ctx);

        while (true) {
            hwctx->lock(hwctx->lock_ctx);
            const HRESULT status = hwctx->device_context->GetData(query, nullptr, 0, 0);
            hwctx->unlock(hwctx->lock_ctx);

            if (status == S_OK) {
                break;
            }
            if (FAILED(status)) {
                JST_ERROR("[REMOTE] Failed to copy the shared frame ({:#x}).", static_cast<U32>(status));
                return Result::ERROR;
            }
            std::this_thread::yield();
        }

        return Result::SUCCESS;
    }

    bool borrowed() const override {
        return false;
    }

 private:
    static int Align(int value) {
        return (value + 15) & ~15;
    }

    AVD3D11VADeviceContext* deviceContext() const {
        const auto* context = reinterpret_cast<const AVHWDeviceContext*>(device->data);
        return static_cast<AVD3D11VADeviceContext*>(context->hwctx);
    }

    Result findAdapter(const SharedDevice& adapter, int& index) {
        IDXGIFactory1* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
            JST_ERROR("[REMOTE] Failed to create the DXGI factory.");
            return Result::ERROR;
        }

        IDXGIAdapter1* candidate = nullptr;
        for (UINT ordinal = 0; index < 0 && factory->EnumAdapters1(ordinal, &candidate) != DXGI_ERROR_NOT_FOUND;
             ++ordinal) {
            DXGI_ADAPTER_DESC1 description = {};
            candidate->GetDesc1(&description);
            U64 luid = 0;
            std::memcpy(&luid, &description.AdapterLuid, sizeof(luid));
            if (luid == adapter.luid) {
                index = static_cast<int>(ordinal);
            }
            candidate->Release();
        }
        factory->Release();

        if (index < 0) {
            JST_ERROR("[REMOTE] No DXGI adapter matches the Vulkan device.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    Result open(const SharedFrame& frame, ID3D11Texture2D*& texture) {
        if (const auto it = textures.find(frame.handle); it != textures.end()) {
            texture = it->second;
            return Result::SUCCESS;
        }

        if (FAILED(device1->OpenSharedResource1(reinterpret_cast<HANDLE>(frame.handle),
                                                __uuidof(ID3D11Texture2D),
                                                reinterpret_cast<void**>(&texture)))) {
            JST_ERROR("[REMOTE] Failed to open the shared frame texture.");
            return Result::ERROR;
        }

        textures[frame.handle] = texture;
        return Result::SUCCESS;
    }

    AVBufferRef* device = nullptr;
    AVBufferRef* frames = nullptr;
    ID3D11Device1* device1 = nullptr;
    ID3D11Query* query = nullptr;
    std::unordered_map<U64, ID3D11Texture2D*> textures;
};

std::unique_ptr<Instance::Remote::Encoder::Input> Instance::Remote::Encoder::CreateD3D11Input() {
    return std::make_unique<D3D11Input>();
}

}  // namespace Jetstream
