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

    auto api = Render::GLES();

    Render::Instance::Config instanceConfig;
    instanceConfig.width = 1920;
    instanceConfig.height = 1080;
    instanceConfig.resizable = true;
    instanceConfig.enableImgui = true;
    instanceConfig.title = "CyberEther";
    auto render = api.createInstance(instanceConfig);

    Render::Program::Config programCfg;
    programCfg.fragmentSource = &fragmentSource;
    programCfg.vertexSource = &vertexSource;
    auto program = api.createProgram(programCfg);

    render->init();

    while (render->keepRunning()) {
        render->clear();

        // Create a window called "My First Tool", with a menu bar.
        ImGui::Begin("My First Tool", NULL, ImGuiWindowFlags_MenuBar);
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */ }
                if (ImGui::MenuItem("Save", "Ctrl+S"))   { /* Do stuff */ }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Plot some values
        const float my_values[] = { 0.2f, 0.1f, 1.0f, 0.5f, 0.9f, 2.2f };
        ImGui::PlotLines("Frame Times", my_values, IM_ARRAYSIZE(my_values));

        // Display contents in a scrolling region
        ImGui::TextColored(ImVec4(1,1,0,1), "Important Stuff");
        ImGui::BeginChild("Scrolling");
        for (int n = 0; n < 50; n++)
            ImGui::Text("%04d: Some text", n);
        ImGui::EndChild();
        ImGui::End();

        render->draw();
        render->step();
    }

    render->terminate();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
