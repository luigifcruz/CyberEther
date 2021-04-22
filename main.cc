#define RENDER_DEBUG

#include "render/base.hpp"
#include "render/extras.hpp"
#include "spectrum/base.hpp"

#include <iostream>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

using R = Render::GLES;
using S = Spectrum::FFTW;

static std::shared_ptr<Render::Instance> render;
static std::shared_ptr<Spectrum::Instance> spectrum;
static std::shared_ptr<Spectrum::LinePlot> lineplot;

void render_loop() {
    spectrum->feed();
    render->start();

    ImGui::Begin("Scene Window");
    ImGui::GetWindowDrawList()->AddImage(
        (void*)lineplot->raw(), ImVec2(ImGui::GetCursorScreenPos()),
        ImVec2(ImGui::GetCursorScreenPos().x + lineplot->config().width/2.0,
        ImGui::GetCursorScreenPos().y + lineplot->config().height/2.0), ImVec2(0, 1), ImVec2(1, 0));
    ImGui::End();

    render->end();
}

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    Render::Instance::Config instanceCfg = {
        .width = 1920,
        .height = 1080,
        .resizable = true,
        .enableImgui = true,
        .enableDebug = true,
        .title = "CyberEther"
    };
    render = Render::Instance::Create<R>(instanceCfg);

    Spectrum::Instance::Config spectrumCfg = {
        .render = render,
        .bandwidth = 10e6,
        .frequency = 96.9e6,
        .size = 0,
        .buffer = nullptr,
        .format = Spectrum::DataFormat::CF64,
    };
    spectrum = Spectrum::Instance::Create<S>(spectrumCfg);

    Spectrum::LinePlot::Config lineplotCfg;
    lineplot = spectrum->create<S>(lineplotCfg);

    render->create();
    spectrum->create();

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_loop, 0, 1);
#else
    while(render->keepRunning())
        render_loop();
#endif

    render->destroy();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
