#define RENDER_DEBUG

#include "render.hpp"
#include "spectrum.hpp"

#include <iostream>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

const GLchar* vertexSource = R"END(#version 300 es
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)END";

const GLchar* mFragmentSource = R"END(#version 300 es
precision highp float;

out vec4 FragColor;

in vec2 TexCoord;

uniform float Scale;
uniform sampler2D ourTexture;

void main() {
    FragColor = texture(ourTexture, TexCoord);
}
)END";

const GLchar* fragmentSource = R"END(#version 300 es
precision highp float;

out vec4 FragColor;

in vec2 TexCoord;

uniform float Scale;
uniform sampler2D ourTexture2;

void main() {
    FragColor = texture(ourTexture2, TexCoord);
}
)END";

int width, height, nrChannels;
unsigned char *data = stbi_load("minime.jpg", &width, &height, &nrChannels, 0);

int width2, height2, nrChannels2;
unsigned char *data2 = stbi_load("bernie", &width2, &height2, &nrChannels2, 0);

auto api = Render::GLES();

Render::Instance::Config instanceCfg;
Render::Surface::Config mSurfaceCfg;
Render::Texture::Config amTextureCfg;
Render::Program::Config mProgramCfg;

std::shared_ptr<Render::GLES::Instance> render;
std::shared_ptr<Render::GLES::Surface> mSurface;
std::shared_ptr<Render::GLES::Texture> amTexture;
std::shared_ptr<Render::GLES::Program> mProgram;

Render::Texture::Config textureCfg;
Render::Surface::Config surfaceCfg;
Render::Texture::Config aTextureCfg;
Render::Program::Config programCfg;

std::shared_ptr<Render::GLES::Texture> texture;
std::shared_ptr<Render::GLES::Surface> surface;
std::shared_ptr<Render::GLES::Texture> aTexture;
std::shared_ptr<Render::GLES::Program> program;

static float scale = 1.0f, scale_min = 0.0f, scale_max = 1.0f;

void render_loop() {
    render->clear();

    mProgram->setUniform("Scale", std::vector<float>{ scale });

    // Create a window called "My First Tool", with a menu bar.
    ImGui::Begin("My First Tool", NULL, ImGuiWindowFlags_MenuBar);
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */ }
            if (ImGui::MenuItem("Save", "Ctrl+S"))   {
            }
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
        (void*)texture->raw(), ImVec2(ImGui::GetCursorScreenPos()),
        ImVec2(ImGui::GetCursorScreenPos().x + width2/2.0, ImGui::GetCursorScreenPos().y + height2/2.0), ImVec2(0, 1), ImVec2(1, 0));

    ImGui::End();

    render->draw();
    render->step();
}

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    instanceCfg.width = width;
    instanceCfg.height = height;
    instanceCfg.resizable = true;
    instanceCfg.enableImgui = true;
    instanceCfg.title = "CyberEther";
    render = api.createInstance(instanceCfg);

    mSurfaceCfg.width = &instanceCfg.width;
    mSurfaceCfg.height = &instanceCfg.height;
    mSurface = api.createSurface(mSurfaceCfg);
    
    amTextureCfg.height = height;
    amTextureCfg.width = width;
    amTextureCfg.buffer = data;
    amTexture = api.createTexture(amTextureCfg);
    
    mProgramCfg.fragmentSource = &mFragmentSource;
    mProgramCfg.vertexSource = &vertexSource;
    mProgramCfg.surface = mSurface;
    mProgramCfg.textures = Render::TexturePlan({
                {"ourTexture", amTexture}
            });
    mProgram = api.createProgram(mProgramCfg);

    textureCfg.height = height2;
    textureCfg.width = width2;
    texture = api.createTexture(textureCfg);

    surfaceCfg.width = &textureCfg.width;
    surfaceCfg.height = &textureCfg.height;
    surfaceCfg.texture = texture;
    surface = api.createSurface(surfaceCfg);

    aTextureCfg.height = height2;
    aTextureCfg.width = width2;
    aTextureCfg.buffer = data2;
    aTexture = api.createTexture(aTextureCfg);

    programCfg.fragmentSource = &fragmentSource;
    programCfg.vertexSource = &vertexSource;
    programCfg.surface = surface;
    programCfg.textures = Render::TexturePlan({
                {"ourTexture", aTexture}
            });
    program = api.createProgram(programCfg);

    render->init();

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_loop, 0, 1);
#else
    while(render->keepRunning())
        render_loop();
#endif

    render->terminate();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
