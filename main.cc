#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <thread>

#include "render/base.hpp"
#include "samurai/samurai.hpp"
#include "jetstream/base.hh"

using namespace Jetstream;
class UI {
public:
    explicit UI() {
        // Configure Render
        Render::Instance::Config renderCfg;
        renderCfg.size = {3130, 1140};
        renderCfg.resizable = true;
        renderCfg.imgui = true;
        renderCfg.vsync = true;
        renderCfg.debug = true;
        renderCfg.title = "CyberEther";
        Render::Init(Render::Backend::GLES, renderCfg);

        // Allocate Radio Buffer
        stream = std::vector<std::complex<float>>(2 << 13);

        // Configure Jetstream
        fft = Jetstream::Block<FFT::Backend<Device::CPU>>({}, {
            Data<VCF32>{Locale::CPU, stream},
        });

        lpt = Jetstream::Block<Lineplot::Backend<Device::CPU>>({}, {
            fft->output(),
        });

        wtf = Jetstream::Block<Waterfall::Backend<Device::CPU>>({}, {
            fft->output(),
        });

        Jetstream::Stream({fft, lpt, wtf});
    }

    bool keep_running() {
        return Render::KeepRunning();
    }

    void begin() {
        Render::Create();

        dsp = std::thread([&]{
            device = std::make_shared<Samurai::Airspy::Device>();

            deviceConfig.sampleRate = 10e6;
            device->Enable(deviceConfig);

            channelConfig.mode = Samurai::Mode::RX;
            channelConfig.dataFmt = Samurai::Format::F32;
            device->EnableChannel(channelConfig, &rx);

            channelState.enableAGC = true;
            channelState.frequency = 96.9e6;
            device->UpdateChannel(rx, channelState);

            device->StartStream();

            streaming = true;
            while (streaming) {
                device->ReadStream(rx, stream.data(), stream.size(), 1000);
                Jetstream::Compute();
            }

            device->StopStream();
        });
    }

    void stop() {
        streaming = false;
        dsp.join();

        Render::Destroy();
    }

    ImVec2 GetRelativeMousePos() {
        ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
        ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
        return ImVec2(mousePositionAbsolute.x - screenPositionAbsolute.x,
                      mousePositionAbsolute.y - screenPositionAbsolute.y);
    }

    void render_step() {
        Render::Begin();

        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        {
            ImGui::Begin("Lineplot");
            auto [x, y] = ImGui::GetContentRegionAvail();
            auto [width, height] = lpt->size({(int)x, (int)y});
            ImGui::Image(lpt->tex().raw(), ImVec2(width, height));
            ImGui::End();
        }

        {
            ImGui::Begin("Waterfall");
            auto [x, y] = ImGui::GetContentRegionAvail();
            auto [width, height] = wtf->size({(int)x, (int)y});
            ImGui::Image(wtf->tex().raw(), ImVec2(width, height));

            if (ImGui::IsItemHovered() && ImGui::IsAnyMouseDown()) {
                if (position == 0) {
                    position = (GetRelativeMousePos().x / wtf->zoom()) + wtf->offset();
                }
                wtf->offset(position - (GetRelativeMousePos().x / wtf->zoom()));
            } else {
                position = 0;
            }

            ImGui::End();
        }

        {
            ImGui::Begin("Control");

            ImGui::InputFloat("Frequency (Hz)", &channelState.frequency);
            if (ImGui::Button("Tune")) {
                device->UpdateChannel(rx, channelState);
            }

            auto [min, max] = fft->amplitude();
            if (ImGui::DragFloatRange2("dBFS Range", &min, &max,
                        1, -300, 0, "Min: %.0f dBFS", "Max: %.0f dBFS")) {
                fft->amplitude({min, max});
            }

            auto interpolate = wtf->interpolate();
            if (ImGui::Checkbox("Interpolate Waterfall", &interpolate)) {
                wtf->interpolate(interpolate);
            }

            auto zoom = wtf->zoom();
            if (ImGui::DragFloat("Waterfall Zoom", &zoom, 0.01, 1.0, 5.0, "%f", 0)) {
                wtf->zoom(zoom);
            }

            ImGui::End();
        }

        ImGui::Begin("Samurai Info");
        if (streaming) {
            float bufferUsageRatio = (float)device->BufferOccupancy(rx) /
                (float)device->BufferCapacity(rx);
            ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text("Buffer Usage");
        }
        ImGui::End();

        Render::Synchronize();
        Jetstream::Present();
        Render::End();
    }

private:
    std::thread dsp;
    bool streaming = false;

    int position;

    // Jetstream
    std::shared_ptr<Jetstream::FFT::Generic> fft;
    std::shared_ptr<Jetstream::Lineplot::Generic> lpt;
    std::shared_ptr<Jetstream::Waterfall::Generic> wtf;

    // Samurai
    Samurai::ChannelId rx;
    Samurai::Channel::Config channelConfig;
    Samurai::Device::Config deviceConfig;
    Samurai::Channel::State channelState;
    std::shared_ptr<Samurai::Device> device;
    std::vector<std::complex<float>> stream;
};

auto ui = UI();

void main_loop() {
    ui.render_step();
}


int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    ui.begin();

    #ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(main_loop, 0, 1);
    #else
    while (ui.keep_running()) {
        ui.render_step();
    }
    #endif

    ui.stop();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
