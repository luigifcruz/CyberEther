#include <thread>

#include "jetstream/base.hh"
#include "samurai/samurai.hpp"

using namespace Jetstream;

class UI {
public:
    explicit UI() {
        // Initialize Backend
        Backend::Initialize<Device::Metal>({});

        // Initialize Render
        Render::Window::Config renderCfg;
        renderCfg.size = {3130, 1140};
        renderCfg.resizable = true;
        renderCfg.imgui = true;
        renderCfg.vsync = true;
        renderCfg.title = "CyberEther";
        Render::Initialize<Device::Metal>(renderCfg);

        // Allocate Radio Buffer
        stream = std::make_unique<Memory::Vector<Device::CPU, CF32>>(2 << 18);

        // Configure Jetstream
        win = Block<Window, Device::CPU>({
            .size = stream->size(),
        }, {});

        mul = Block<Multiply, Device::CPU>({
            .size = stream->size(),
        }, {
            .factorA = *stream,
            .factorB = win->getWindowBuffer(),
        });

        fft = Block<FFT, Device::CPU>({
            .size = stream->size(),
        }, {
            .buffer = mul->getProductBuffer(),
        });

        amp = Block<Amplitude, Device::CPU>({
            .size = stream->size(),
        }, {
            .buffer = fft->getOutputBuffer(),
        });

        scl = Block<Scale, Device::CPU>({
            .size = stream->size(),
            .range = {-100.0, 0.0},
        }, {
            .buffer = amp->getOutputBuffer(),
        });

        lpt = Block<Lineplot, Device::CPU>({}, {
            .buffer = scl->getOutputBuffer(),
        });
    }

    bool keep_running() {
        return Render::KeepRunning();
    }

    void begin() {
        Render::Create();

        dsp = std::thread([&]{
            device = std::make_shared<Samurai::LimeSDR::Device>();

            deviceConfig.sampleRate = 30e6;
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
                device->ReadStream(rx, stream->data(), stream->size(), 1000);
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

        {
            ImGui::Begin("Lineplot");
            auto [x, y] = ImGui::GetContentRegionAvail();
            auto [width, height] = lpt->size({(U64)x, (U64)y});
            ImGui::Image(lpt->getTexture().raw(), ImVec2(width, height));
            ImGui::End();
        }

        {
            ImGui::Begin("Control");

            ImGui::InputFloat("Frequency (Hz)", &channelState.frequency);
            if (ImGui::Button("Tune")) {
                device->UpdateChannel(rx, channelState);
            }

            auto [min, max] = scl->range();
            if (ImGui::DragFloatRange2("dBFS Range", &min, &max,
                        1, -300, 0, "Min: %.0f dBFS", "Max: %.0f dBFS")) {
                scl->range({min, max});
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
    std::shared_ptr<Window<Device::CPU>> win;
    std::shared_ptr<Multiply<Device::CPU>> mul;
    std::shared_ptr<FFT<Device::CPU>> fft;
    std::shared_ptr<Amplitude<Device::CPU>> amp;
    std::shared_ptr<Scale<Device::CPU>> scl;
    std::shared_ptr<Lineplot<Device::CPU>> lpt;

    // Samurai
    Samurai::ChannelId rx;
    Samurai::Channel::Config channelConfig;
    Samurai::Device::Config deviceConfig;
    Samurai::Channel::State channelState;
    std::shared_ptr<Samurai::Device> device;
    std::unique_ptr<Memory::Vector<Device::CPU, CF32>> stream;
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
