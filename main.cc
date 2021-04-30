#define RENDER_DEBUG

#include <iostream>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "render/base.hpp"
#include "samurai/samurai.hpp"
#include "jetstream/fft/base.hpp"
#include "jetstream/lineplot/base.hpp"

namespace T = Jetstream::DF::CPU;

struct State {
    bool streaming = false;

    Samurai::ChannelId rx;
    std::shared_ptr<Render::Instance> render;
    std::shared_ptr<Samurai::Airspy::Device> device;

    T::CF32V fftDf;
    std::shared_ptr<Jetstream::FFT::Generic> fft;

    T::CF32V lptDf;
    std::shared_ptr<Jetstream::Lineplot::Generic> lpt;
};

void dsp_loop(std::shared_ptr<State> state) {
    while (state->streaming) {
        state->device->ReadStream(state->rx, state->fftDf.input->data(), state->fftDf.input->size(), 1000);
        JETSTREAM_ASSERT_SUCCESS(state->fft->compute());
        JETSTREAM_ASSERT_SUCCESS(state->lpt->compute(state->fft));
        JETSTREAM_ASSERT_SUCCESS(state->lpt->barrier())
    }
}

void render_loop(std::shared_ptr<State> state) {
    state->render->start();

    state->lpt->present();

    ImGui::Begin("Lineplot");
    ImGui::GetWindowDrawList()->AddImage(
        (void*)state->lpt->tex()->raw(), ImVec2(ImGui::GetCursorScreenPos()),
        ImVec2(ImGui::GetCursorScreenPos().x + state->lpt->conf().width/2.0,
        ImGui::GetCursorScreenPos().y + state->lpt->conf().height/2.0), ImVec2(0, 1), ImVec2(1, 0));
    ImGui::End();

    ImGui::Begin("Samurai Info");
    ImGui::Text("Buffer Fill: %ld", state->device->BufferOccupancy(state->rx));
    ImGui::End();

    state->render->end();
}

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    auto state = std::make_shared<State>();

    Render::Instance::Config renderCfg;
    renderCfg.width = 3000;
    renderCfg.height = 1080;
    renderCfg.resizable = true;
    renderCfg.enableImgui = true;
    renderCfg.enableDebug = true;
    renderCfg.enableVsync = true;
    renderCfg.title = "CyberEther";
    state->render = Render::Instantiate(Render::API::GLES, renderCfg);

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

    Jetstream::FFT::Config fftCfg;
    state->fftDf.input = std::make_shared<std::vector<std::complex<float>>>(8192*8);
    state->fft = Jetstream::FFT::Instantiate(fftCfg, state->fftDf);

    Jetstream::Lineplot::Config lptCfg{state->render};
    state->lptDf.input = state->fftDf.output;
    state->lpt = Jetstream::Lineplot::Instantiate(lptCfg, state->lptDf);

    state->render->create();
    state->device->StartStream();

    state->streaming = true;
    std::thread dsp(dsp_loop, state);

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_loop, 0, 1);
#else
    while(state->render->keepRunning())
        render_loop(state);
#endif

    state->streaming = false;
    dsp.join();

    state->device->StopStream();
    state->render->destroy();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
