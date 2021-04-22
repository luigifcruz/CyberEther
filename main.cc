#define RENDER_DEBUG

#include "render/base.hpp"
#include "render/extras.hpp"
#include "spectrum/base.hpp"

#include <iostream>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using T = Render::GLES;

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
uniform int vertexIdx;
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
uniform int vertexIdx;
uniform sampler2D ourTexture2;

void main() {
    if (vertexIdx == 1) {
        FragColor = vec4(Scale, 1.0, 0.0, 0.0);
    } else {
        FragColor = texture(ourTexture2, TexCoord);
    }
}
)END";

int width, height, nrChannels;
unsigned char *data = stbi_load("minime.jpg", &width, &height, &nrChannels, 0);

int width2, height2, nrChannels2;
unsigned char *data2 = stbi_load("bernie", &width2, &height2, &nrChannels2, 0);

Render::Instance::Config instanceCfg;
Render::Surface::Config mSurfaceCfg;
Render::Texture::Config amTextureCfg;
Render::Program::Config mProgramCfg;
Render::Vertex::Config mVertexCfg;

std::shared_ptr<Render::Instance> render;
std::shared_ptr<Render::Surface> mSurface;
std::shared_ptr<Render::Texture> amTexture;
std::shared_ptr<Render::Program> mProgram;
std::shared_ptr<Render::Vertex> mVertex;

Render::Texture::Config textureCfg;
Render::Surface::Config surfaceCfg;
Render::Texture::Config aTextureCfg;
Render::Program::Config programCfg;
Render::Vertex::Config vertexCfg;

std::shared_ptr<Render::Texture> texture;
std::shared_ptr<Render::Surface> surface;
std::shared_ptr<Render::Texture> aTexture;
std::shared_ptr<Render::Program> program;
std::shared_ptr<Render::Vertex> vertex;
std::shared_ptr<Render::Vertex> vertexb;

static float scale = -1.0f, scale_min = -1.0f, scale_max = 1.0f;

void render_loop() {
    render->start();

    vertexCfg.buffers.at(0).data[1] = scale;
    program->setUniform("Scale", std::vector<float>{ scale });
    vertex->update();

    ImGui::ShowMetricsWindow();

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
    ImGui::Text("Value: %f", vertexCfg.buffers.at(0).data[1]);
    ImGui::End();

    ImGui::Begin("Render Info");
    ImGui::Text("Renderer Name: %s", render->renderer_str().c_str());
    ImGui::Text("Renderer Vendor: %s", render->vendor_str().c_str());
    ImGui::Text("Renderer Version: %s", render->version_str().c_str());
    ImGui::Text("Renderer GLSL Version: %s", render->glsl_str().c_str());
    ImGui::End();

    ImGui::Begin("Scene Window");

    ImGui::GetWindowDrawList()->AddImage(
        (void*)texture->raw(), ImVec2(ImGui::GetCursorScreenPos()),
        ImVec2(ImGui::GetCursorScreenPos().x + width2/2.0, ImGui::GetCursorScreenPos().y + height2/2.0), ImVec2(0, 1), ImVec2(1, 0));

    ImGui::End();

    render->end();
}

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    static std::vector<float> a;
    static std::vector<uint> p;
    int b = 0;
    for (float i = -1.0f; i < +1.0f; i += 0.10f) {
        a.push_back(-1.0f);
        a.push_back(i);
        a.push_back(+0.0f);
        p.push_back(b++);
        a.push_back(+1.0f);
        a.push_back(i);
        a.push_back(+0.0f);
        p.push_back(b++);
        a.push_back(i);
        a.push_back(-1.0f);
        a.push_back(+0.0f);
        p.push_back(b++);
        a.push_back(i);
        a.push_back(+1.0f);
        a.push_back(+0.0f);
        p.push_back(b++);
    }

    instanceCfg.width = width;
    instanceCfg.height = height;
    instanceCfg.resizable = true;
    instanceCfg.enableImgui = true;
    instanceCfg.title = "CyberEther";
    render = Render::Instance::Create<T>(instanceCfg);

    {
        mVertexCfg.buffers = Render::Extras::FillScreenVertices;
        mVertexCfg.indices = Render::Extras::FillScreenIndices;
        mVertex = render->create<T>(mVertexCfg);

        amTextureCfg.height = height;
        amTextureCfg.width = width;
        amTextureCfg.buffer = data;
        amTextureCfg.key = "ourTexture";
        amTexture = render->create<T>(amTextureCfg);

        mProgramCfg.fragmentSource = &mFragmentSource;
        mProgramCfg.vertexSource = &vertexSource;
        mProgramCfg.vertices = {mVertex};
        mProgramCfg.textures = {amTexture};
        mProgram = render->create<T>(mProgramCfg);

        mSurfaceCfg.width = &instanceCfg.width;
        mSurfaceCfg.height = &instanceCfg.height;
        mSurfaceCfg.programs = {mProgram};
        mSurface = render->create<T>(mSurfaceCfg);
    }

    {
        vertexb = render->create<T>(mVertexCfg);

        vertexCfg.indices = p;
        vertexCfg.buffers = {
            {
                .data = a.data(),
                .size = a.size(),
                .stride = 3,
                .usage = Render::Vertex::Buffer::Dynamic,
            },
        };
        vertexCfg.mode = Render::Vertex::Mode::Lines;
        vertex = render->create<T>(vertexCfg);

        aTextureCfg.height = height2;
        aTextureCfg.width = width2;
        aTextureCfg.buffer = data2;
        aTextureCfg.key = "ourTexture2";
        aTexture = render->create<T>(aTextureCfg);

        programCfg.fragmentSource = &fragmentSource;
        programCfg.vertexSource = &vertexSource;
        programCfg.vertices = {vertexb, vertex};
        programCfg.textures = {aTexture};
        program = render->create<T>(programCfg);

        textureCfg.height = height2;
        textureCfg.width = width2;
        texture = render->create<T>(textureCfg);

        surfaceCfg.width = &textureCfg.width;
        surfaceCfg.height = &textureCfg.height;
        surfaceCfg.texture = texture;
        surfaceCfg.programs = {program};
        surface = render->create<T>(surfaceCfg);
    }

    render->create();

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_loop, 0, 1);
#else
    while(render->keepRunning())
        render_loop();
#endif

    render->destroy();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
