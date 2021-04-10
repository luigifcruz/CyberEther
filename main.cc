#define RENDER_DEBUG

#include "render.hpp"
#include "spectrum.hpp"

#include <iostream>

const GLchar* vertexSource = R"END(#version 300 es
layout (location = 0) in vec3 aPos;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
)END";

const GLchar* fragmentSource = R"END(#version 300 es
precision highp float;

out vec4 FragColor;

void main() {
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}
)END";


int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    auto render = Render::instantiate(Render::BackendId::OPENGL);
    auto spectrum = Spectrum::instantiate();

    Render::Config renderCfg;
    renderCfg.title = "Demo App";
    renderCfg.width = 1920;
    renderCfg.height = 1080;
    render->init(renderCfg);

    render->createSurface();
    render->createShaders(vertexSource, fragmentSource);
    render->createImgui();

    while (render->keepRunning()) {
        render->clear();

        ImGui::Begin("Demo window");
        ImGui::Button("Hello!");
        ImGui::End();

        render->draw();
        render->step();
    }

    render->destroyImgui();
    render->destroyShaders();
    render->destroySurface();
    render->terminate();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
