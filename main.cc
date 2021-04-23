#define RENDER_DEBUG

#include "render/base.hpp"
#include "render/extras.hpp"
#include "spectrum/base.hpp"

#include <iostream>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

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

    Render::Instance::Config renderCfg;
    renderCfg.width = 1920;
    renderCfg.height = 1080;
    renderCfg.resizable = true;
    renderCfg.enableImgui = true;
    renderCfg.enableDebug = true;
    renderCfg.title = "CyberEther";
    render = Render::Instantiate(Render::API::GLES, renderCfg);

    Spectrum::Instance::Config spectrumCfg;
    spectrumCfg.render = render;
    spectrum = Spectrum::Instantiate(Spectrum::API::FFTW, spectrumCfg);

    static Spectrum::LinePlot::Config lineplotCfg;
    lineplotCfg.height = 1080;
    lineplotCfg.width = 1920;
    lineplot = spectrum->create(lineplotCfg);

    spectrum->create();
    render->create();

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_loop, 0, 1);
#else
    while(render->keepRunning())
        render_loop();
#endif

    spectrum->destroy();
    render->destroy();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
