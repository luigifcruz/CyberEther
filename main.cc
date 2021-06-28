#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "render/base.hpp"
#include "samurai/samurai.hpp"
#include "jetstream/base.hpp"

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
        renderCfg.title = "CyberEther";
        render = Render::Instantiate(Render::API::GLES, renderCfg);

        // Configure Jetstream Modules
        engine = std::make_shared<Jetstream::Engine>();
        stream = std::vector<std::complex<float>>(2048);

        engine->add<FFT>("fft0", {}, {
            {"input0", Data<VCF32>({Locale::CPU, stream})}
        });

        engine->add<Lineplot>("lpt0", {render}, {
            {"input0", Tap{"fft0", "output0"}},
        });

        engine->add<Waterfall>("wtf0", {render}, {
            {"input0", Tap{"fft0", "output0"}},
        });
    }

    void start() {
        render->create();

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
                JETSTREAM_CHECK_THROW(engine->compute());
            }

            device->StopStream();
        });
    }

    void stop() {
        streaming = false;
        dsp.join();

        render->destroy();
    }

    bool keep_running() {
        return render->keepRunning();
    }

    void render_step() {
        render->start();

        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        auto lpt = engine->get<Lineplot>("lpt0");
        auto wtf = engine->get<Waterfall>("wtf0");
        auto fft = engine->get<FFT>("fft0");

        {
            ImGui::Begin("Lineplot");
            auto regionSize = ImGui::GetContentRegionAvail();
            auto [width, height] = lpt->size({(int)regionSize.x, (int)regionSize.y});
            ImGui::Image((void*)(intptr_t)lpt->tex().lock()->raw(), ImVec2(width, height));
            ImGui::End();
        }

        {
            ImGui::Begin("Waterfall");
            auto regionSize = ImGui::GetContentRegionAvail();
            auto [width, height] = wtf->size({(int)regionSize.x, (int)regionSize.y});
            ImGui::Image((void*)(intptr_t)wtf->tex().lock()->raw(), ImVec2(width, height));
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

        render->synchronize();
        JETSTREAM_CHECK_THROW(engine->present());
        render->end();
    }

private:
    std::thread dsp;
    std::atomic<bool> streaming{false};

    // Render
    std::shared_ptr<Render::Instance> render;

    // Jetstream
    std::shared_ptr<Jetstream::Engine> engine;

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

    ui.start();

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
