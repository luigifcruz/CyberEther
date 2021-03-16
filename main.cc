#define RENDER_DEBUG

#include "render.hpp"
#include "spectrum.hpp"

#include <iostream>

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    auto render = Render::GetRender(Render::Backend::OPENGL);
    if (render->getBackend() == Render::Backend::OPENGL) {
        std::cout << "Correct render picked." << std::endl;
    }

    Render::Config renderCfg;
    renderCfg.title = "Demo App";
    renderCfg.width = 1280;
    renderCfg.height = 720;
    render->init(renderCfg);

    std::cout << "Goodbye from CyberEther!" << std::endl;
}