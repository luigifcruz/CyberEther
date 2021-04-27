#define RENDER_DEBUG

#include "render/base.hpp"
#include "render/extras.hpp"
#include "spectrum/base.hpp"

#include <iostream>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <samurai/samurai.hpp>
#include <complex>
#include <future>

using namespace Samurai;

ChannelId rx;
auto device = Airspy::Device();

static Spectrum::Instance::Config spectrumCfg;
static std::shared_ptr<Render::Instance> render;
static std::shared_ptr<Spectrum::Instance> spectrum;
static std::shared_ptr<Spectrum::LinePlot> lineplot;

void render_loop() {
    ASSERT_SUCCESS(device.ReadStream(rx, spectrumCfg.buffer, spectrumCfg.size, 1000));
    spectrum->feed();
    render->start();

    ImGui::Begin("Scene Window");
    ImGui::GetWindowDrawList()->AddImage(
        (void*)lineplot->raw(), ImVec2(ImGui::GetCursorScreenPos()),
        ImVec2(ImGui::GetCursorScreenPos().x + lineplot->config().width/2.0,
        ImGui::GetCursorScreenPos().y + lineplot->config().height/2.0), ImVec2(0, 1), ImVec2(1, 0));
    ImGui::End();

    ImGui::Begin("Samurai Info");
    ImGui::Text("Buffer Fill: %ld", device.BufferOccupancy(rx));
    ImGui::End();

    render->end();
}

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    Render::Instance::Config renderCfg;
    renderCfg.width = 3000;
    renderCfg.height = 1080;
    renderCfg.resizable = true;
    renderCfg.enableImgui = true;
    renderCfg.enableDebug = true;
    renderCfg.enableVsync = false;
    renderCfg.title = "CyberEther";
    render = Render::Instantiate(Render::API::GLES, renderCfg);

    spectrumCfg.render = render;
    spectrumCfg.bandwidth = 10e6;
    spectrumCfg.frequency = 96.9e6;
    spectrumCfg.size = 8192;
    spectrumCfg.format = Spectrum::DataFormat::CF32;
    spectrumCfg.buffer = (void*)malloc(sizeof(std::complex<float>) * spectrumCfg.size);
    spectrum = Spectrum::Instantiate(Spectrum::API::FFTW, spectrumCfg);

    Spectrum::LinePlot::Config lineplotCfg;
    lineplotCfg.height = 1080;
    lineplotCfg.width = 4000;
    lineplot = spectrum->create(lineplotCfg);

    Device::Config deviceConfig{};
    deviceConfig.sampleRate = 10e6;
    device.Enable(deviceConfig);

    Channel::Config channelConfig{};
    channelConfig.mode = Mode::RX;
    channelConfig.dataFmt = Format::F32;
    ASSERT_SUCCESS(device.EnableChannel(channelConfig, &rx));

    Channel::State channelState{};
    channelState.enableAGC = true;
    channelState.frequency = 96.9e6;
    ASSERT_SUCCESS(device.UpdateChannel(rx, channelState));

    spectrum->create();
    render->create();
    device.StartStream();

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_loop, 0, 1);
#else
    while(render->keepRunning())
        render_loop();
#endif

    device.StopStream();
    spectrum->destroy();
    render->destroy();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
