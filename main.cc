#include <thread>
#include <chrono>

#include "jetstream/base.hh"
#include "samurai/samurai.hpp"

using namespace Jetstream;

using ST = Memory::Vector<Device::CPU, CF32>;

class DSP {
 public:
    DSP() {
        stream = std::make_shared<Memory::Vector<Device::CPU, CF32>>(2 << 11);
        streaming = true;
        worker = std::thread([&]{ this->threadLoop(); });
    }

    ~DSP() {
        streaming = false;
        worker.join();

        JST_DEBUG("The DSP was destructed.");
    }

    constexpr const ST& getOutput() const {
        return *stream;
    }

    constexpr const F32 getBufferCapacity() const {
        return device->BufferCapacity(rx);
    }

    constexpr const F32 getBufferOccupancy() const {
        return device->BufferOccupancy(rx);
    }

    constexpr F32& getTunerFrequency() {
        return channelState.frequency;
    }

    constexpr void setTunerFrequency() {
        device->UpdateChannel(rx, channelState);
    }

 private:
    std::thread worker;

    bool streaming = false;
    std::shared_ptr<ST> stream;

    Samurai::ChannelId rx;
    Samurai::Channel::Config channelConfig;
    Samurai::Device::Config deviceConfig;
    Samurai::Channel::State channelState;
    std::shared_ptr<Samurai::Device> device;

    void threadLoop() {
        while (streaming) {
            device = std::make_shared<Samurai::Airspy::Device>();

            deviceConfig.sampleRate = 10e6;
            device->Enable(deviceConfig);

            channelConfig.mode = Samurai::Mode::RX;
            channelConfig.dataFmt = Samurai::Format::F32;
            device->EnableChannel(channelConfig, &rx);

            channelState.enableAGC = true;
            channelState.frequency = 112.9e6;
            device->UpdateChannel(rx, channelState);

            device->StartStream();

            streaming = true;
            while (streaming) {
                device->ReadStream(rx, stream->data(), stream->size(), 1000);
                Jetstream::Compute();
            }

            device->StopStream();
        }
    }
};

class UI {
 public:
    UI(DSP& dsp) : dsp(dsp) {
        auto& stream = dsp.getOutput();

        // Initialize Render
        Render::Window::Config renderCfg;
        renderCfg.size = {3130, 1140};
        renderCfg.resizable = true;
        renderCfg.imgui = true;
        renderCfg.vsync = true;
        renderCfg.title = "CyberEther";
        Render::Initialize<Device::Metal>(renderCfg);

        // Configure Jetstream
        win = Block<Window, Device::CPU>({
            .size = stream.size(),
        }, {});

        mul = Block<Multiply, Device::CPU>({
            .size = stream.size(),
        }, {
            .factorA = stream,
            .factorB = win->getWindowBuffer(),
        });

        fft = Block<FFT, Device::CPU>({
            .size = stream.size(),
        }, {
            .buffer = mul->getProductBuffer(),
        });

        amp = Block<Amplitude, Device::CPU>({
            .size = stream.size(),
        }, {
            .buffer = fft->getOutputBuffer(),
        });

        scl = Block<Scale, Device::CPU>({
            .size = stream.size(),
            .range = {-100.0, 0.0},
        }, {
            .buffer = amp->getOutputBuffer(),
        });

        lpt = Block<Lineplot, Device::CPU>({}, {
            .buffer = scl->getOutputBuffer(),
        });

        wtf = Block<Waterfall, Device::CPU>({}, {
            .buffer = scl->getOutputBuffer(),
        });

        Render::Create();

        streaming = true;
        worker = std::thread([&]{ this->threadLoop(); });
    }

    ~UI() {
        streaming = false;
        worker.join();

        Render::Destroy();

        JST_DEBUG("The UI was destructed.");
    }

 private:
    DSP& dsp;
    std::thread worker;

    bool streaming = false;

    std::shared_ptr<Window<Device::CPU>> win;
    std::shared_ptr<Multiply<Device::CPU>> mul;
    std::shared_ptr<FFT<Device::CPU>> fft;
    std::shared_ptr<Amplitude<Device::CPU>> amp;
    std::shared_ptr<Scale<Device::CPU>> scl;
    std::shared_ptr<Lineplot<Device::CPU>> lpt;
    std::shared_ptr<Waterfall<Device::CPU>> wtf;

    ImVec2 getRelativeMousePos() {
        ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
        ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
        return ImVec2(mousePositionAbsolute.x - screenPositionAbsolute.x,
                      mousePositionAbsolute.y - screenPositionAbsolute.y);
    }

    void threadLoop() {
        int position;

        while (streaming && Render::KeepRunning()) {
            Render::Begin();

            ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

            {
                ImGui::Begin("Waterfall");

                auto [x, y] = ImGui::GetContentRegionAvail();
                auto [width, height] = wtf->viewSize({(U64)x, (U64)y});
                ImGui::Image(wtf->getTexture().raw(), ImVec2(width, height));

                if (ImGui::IsItemHovered() && ImGui::IsAnyMouseDown()) {
                    if (position == 0) {
                        position = (getRelativeMousePos().x / wtf->zoom()) + wtf->offset();
                    }
                    wtf->offset(position - (getRelativeMousePos().x / wtf->zoom()));
                } else {
                    position = 0;
                }

                ImGui::End();
            }

            {
                ImGui::Begin("Lineplot");
                
                auto [x, y] = ImGui::GetContentRegionAvail();
                auto [width, height] = lpt->viewSize({(U64)x, (U64)y});
                ImGui::Image(lpt->getTexture().raw(), ImVec2(width, height));

                ImGui::End();
            }

            {
                ImGui::Begin("Control");

                ImGui::InputFloat("Frequency (Hz)", &dsp.getTunerFrequency());
                if (ImGui::Button("Tune")) {
                    dsp.setTunerFrequency();
                }

                auto [min, max] = scl->range();
                if (ImGui::DragFloatRange2("dBFS Range", &min, &max,
                            1, -300, 0, "Min: %.0f dBFS", "Max: %.0f dBFS")) {
                    scl->range({min, max});
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

            {
                ImGui::Begin("Samurai Info");

                if (streaming) {
                    float bufferUsageRatio = dsp.getBufferOccupancy() / 
                                             dsp.getBufferCapacity();
                    ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
                    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
                    ImGui::Text("Buffer Usage");
                }

                ImGui::End();
            }

            Render::Synchronize();
            Jetstream::Present();
            Render::End();
        }
    }
};

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    Backend::Initialize<Device::Metal>({});

    {
        auto dsp = DSP();
        auto ui = UI(dsp);

        while (Render::KeepRunning()) {
            Render::PollEvents();
        }
    }

    Backend::Destroy<Device::Metal>();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
