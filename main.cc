#define RENDER_DEBUG

#include "render/base.hpp"
#include "samurai/samurai.hpp"
#include "jetstream/fft/base.hpp"
#include "jetstream/lineplot/base.hpp"

struct State {
    bool streaming = false;

    // Render
    std::shared_ptr<Render::Instance> render;


    // Samurai
    Samurai::ChannelId rx;
    std::shared_ptr<Samurai::Airspy::Device> device;

    // Jetstream
    std::vector<std::complex<float>> stream;
    std::vector<std::shared_ptr<Jetstream::Module>> modules;
};

auto state = std::make_shared<State>();

void dsp_loop(std::shared_ptr<State> state) {
    while (state->streaming) {
        state->device->ReadStream(state->rx, state->stream.data(), state->stream.size(), 1000);
        JETSTREAM_ASSERT_SUCCESS(Jetstream::Compute(state->modules));
        JETSTREAM_ASSERT_SUCCESS(Jetstream::Barrier(state->modules));
    }
}

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    // Configure Render
    Render::Instance::Config renderCfg;
    renderCfg.width = 3000;
    renderCfg.height = 1080;
    renderCfg.resizable = true;
    renderCfg.enableImgui = true;
    renderCfg.enableDebug = true;
    renderCfg.enableVsync = true;
    renderCfg.title = "CyberEther";
    state->render = Render::Instantiate(Render::API::GLES, renderCfg);

    // Configure Samurai Radio
    state->device = std::make_shared<Samurai::Airspy::Device>();

    Samurai::Device::Config deviceConfig;
    deviceConfig.sampleRate = 10e6;
    state->device->Enable(deviceConfig);

    Samurai::Channel::Config channelConfig;
    channelConfig.mode = Samurai::Mode::RX;
    channelConfig.dataFmt = Samurai::Format::F32;
    state->device->EnableChannel(channelConfig, &state->rx);

    Samurai::Channel::State channelState;
    channelState.enableAGC = true;
    channelState.frequency = 545.5e6;
    state->device->UpdateChannel(state->rx, channelState);

    // Configure Jetstream Modules
    auto device = Jetstream::Locale::CPU;
    state->stream = std::vector<std::complex<float>>(8192*8);

    Jetstream::FFT::Config fftCfg;
    fftCfg.input0 = {Jetstream::Locale::CPU, state->stream};
    fftCfg.policy = {Jetstream::Policy::ASYNC, {}};
    auto fft = Jetstream::FFT::Instantiate(device, fftCfg);

    Jetstream::Lineplot::Config lptCfg;
    lptCfg.render = state->render;
    lptCfg.input0 = fft->output();
    lptCfg.policy = {Jetstream::Policy::ASYNC, {fft}};
    auto lpt = Jetstream::Lineplot::Instantiate(device, lptCfg);

    // Add Jetstream modules to the execution pipeline.
    state->modules.push_back(fft);
    state->modules.push_back(lpt);

    // Start Components
    state->streaming = true;
    state->render->create();
    state->device->StartStream();
    std::thread dsp(dsp_loop, state);

    while(state->render->keepRunning()) {
        state->render->start();

        JETSTREAM_ASSERT_SUCCESS(Jetstream::Present(state->modules));

        ImGui::Begin("Lineplot");
        ImGui::GetWindowDrawList()->AddImage(
            (void*)lpt->tex()->raw(), ImVec2(ImGui::GetCursorScreenPos()),
            ImVec2(ImGui::GetCursorScreenPos().x + lpt->conf().width/2.0,
            ImGui::GetCursorScreenPos().y + lpt->conf().height/2.0), ImVec2(0, 1), ImVec2(1, 0));
        ImGui::End();

        ImGui::Begin("Samurai Info");
        ImGui::Text("Buffer Fill: %ld", state->device->BufferOccupancy(state->rx));
        ImGui::End();

        state->render->end();
    }

    state->streaming = false;
    dsp.join();

    state->device->StopStream();
    state->render->destroy();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
