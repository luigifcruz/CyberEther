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

const GLchar* mFragmentSource = R"END(#version 300 es
precision highp float;

out vec4 FragColor;

void main() {
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}
)END";

const GLchar* fragmentSource = R"END(#version 300 es
precision highp float;

out vec4 FragColor;

uniform float Scale;

void main() {
    FragColor = vec4(1.0f, Scale, 1.0f, 1.0f);
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

    Render::Surface::Config mSurfaceCfg;
    mSurfaceCfg.height = 1080;
    mSurfaceCfg.width = 1920;
    mSurfaceCfg.default_s = true;
    auto mSurface = api.createSurface(mSurfaceCfg);

    Render::Program::Config mProgramCfg;
    mProgramCfg.fragmentSource = &mFragmentSource;
    mProgramCfg.vertexSource = &vertexSource;
    mProgramCfg.surface = mSurface;
    auto mProgram = api.createProgram(mProgramCfg);

    Render::Surface::Config surfaceCfg;
    surfaceCfg.height = 1080;
    surfaceCfg.width = 1920;
    surfaceCfg.default_s = false;
    auto surface = api.createSurface(surfaceCfg);

    Render::Program::Config programCfg;
    programCfg.fragmentSource = &fragmentSource;
    programCfg.vertexSource = &vertexSource;
    programCfg.surface = surface;
    auto program = api.createProgram(programCfg);

    render->init();

    static float scale = 1.0f, scale_min = 0.0f, scale_max = 1.0f;

    while (render->keepRunning()) {
        render->clear();

        program->setUniform("Scale", std::vector<float>{ scale });

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

        ImGui::Begin("Uniform Tester");
        ImGui::SliderScalar("range", ImGuiDataType_Float, &scale, &scale_min, &scale_max);
        ImGui::End();

        ImGui::Begin("Scene Window");

        ImGui::GetWindowDrawList()->AddImage(
            surface->getRawTexture(), ImVec2(ImGui::GetCursorScreenPos()),
            ImVec2(ImGui::GetCursorScreenPos().x + instanceConfig.width/2.0, ImGui::GetCursorScreenPos().y + instanceConfig.height/2.0), ImVec2(0, 1), ImVec2(1, 0));

        ImGui::End();

        render->draw();
        render->step();
    }

    render->terminate();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
