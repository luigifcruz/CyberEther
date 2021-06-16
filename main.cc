#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "render/base.hpp"
#include "samurai/samurai.hpp"
#include "jetstream/fft/base.hpp"
#include "jetstream/lineplot/base.hpp"
#include "jetstream/waterfall/base.hpp"
#include "jetstream/histogram/base.hpp"

class UI {
public:
    explicit UI() {
        // Configure Render
        renderCfg.width = 3130;
        renderCfg.height = 1140;
        renderCfg.resizable = true;
        renderCfg.enableImgui = true;
        renderCfg.enableVsync = true;
        renderCfg.title = "CyberEther";
        render = Render::Instantiate(Render::API::GLES, renderCfg);

        // Configure Jetstream Modules
        auto device = Jetstream::Locale::CPU;
        engine = std::make_shared<Jetstream::Engine>();
        stream = std::vector<std::complex<float>>(2048);

        fftCfg.input0 = {Jetstream::Locale::CPU, stream};
        fftCfg.policy = {Jetstream::Launch::ASYNC, {}};
        fft = Jetstream::FFT::Instantiate(device, fftCfg);

        lptCfg.render = render;
        lptCfg.input0 = fft->output();
        lptCfg.policy = {Jetstream::Launch::ASYNC, {fft}};
        lpt = Jetstream::Lineplot::Instantiate(device, lptCfg);

        wtfCfg.render = render;
        wtfCfg.input0 = fft->output();
        wtfCfg.policy = {Jetstream::Launch::ASYNC, {fft}};
        wtf = Jetstream::Waterfall::Instantiate(device, wtfCfg);

        // Add Jetstream modules to the execution pipeline.
        engine->push_back(fft);
        engine->push_back(lpt);
        engine->push_back(wtf);
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
                JETSTREAM_ASSERT_SUCCESS(engine->compute());
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

        JETSTREAM_ASSERT_SUCCESS(engine->present());

        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        ImGui::Begin("Lineplot");
        lptCfg.width = ImGui::GetContentRegionAvail().x;
        lptCfg.height = ImGui::GetContentRegionAvail().y;
        ImGui::Image((void*)(intptr_t)lpt->tex()->raw(), ImVec2(lpt->conf().width, lpt->conf().height));
        ImGui::End();

        ImGui::Begin("Waterfall");
        wtfCfg.width = ImGui::GetContentRegionAvail().x;
        wtfCfg.height = ImGui::GetContentRegionAvail().y;
        ImGui::Image((void*)(intptr_t)wtf->tex()->raw(), ImVec2(wtf->conf().width, wtf->conf().height));
        ImGui::End();

        ImGui::Begin("Control");
        ImGui::InputFloat("Frequency (Hz)", &channelState.frequency);
        if (ImGui::Button("Tune")) {
            device->UpdateChannel(rx, channelState);
        }
        ImGui::DragFloatRange2("dBFS Range", &fftCfg.min_db, &fftCfg.max_db,
             1, -300, 0, "Min: %.0f dBFS", "Max: %.0f dBFS");
        ImGui::Checkbox("Interpolate Waterfall", &wtfCfg.interpolate);
        ImGui::End();

        ImGui::Begin("Samurai Info");
        if (streaming) {
            float bufferUsageRatio = (float)device->BufferOccupancy(rx) /
                (float)device->BufferCapacity(rx);
            ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text("Buffer Usage");
        }
        ImGui::End();

        render->end();
    }

private:
    std::thread dsp;
    std::atomic<bool> streaming{false};

    // Render
    Render::Instance::Config renderCfg;
    std::shared_ptr<Render::Instance> render;

    // Jetstream
    Jetstream::FFT::Config fftCfg;
    Jetstream::Lineplot::Config lptCfg;
    Jetstream::Waterfall::Config wtfCfg;
    std::shared_ptr<Jetstream::Engine> engine;
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
