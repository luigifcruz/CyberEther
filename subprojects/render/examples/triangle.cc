#include "render/extras.hpp"
#define RENDER_DEBUG

#include "render/base.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

const char* shaders = R"END(
    #include <metal_stdlib>

    using namespace metal;

    struct TexturePipelineRasterizerData {
        float4 position [[position]];
        float2 texcoord;
    };

    vertex TexturePipelineRasterizerData vertFunc(
        const device packed_float3* vertexArray [[buffer(0)]],
        const device packed_float2* texcoord [[buffer(1)]],
        unsigned int vID[[vertex_id]]) {
        TexturePipelineRasterizerData out;

        out.position = vector_float4(vertexArray[vID], 1.0);
        out.texcoord = 1 - texcoord[vID];

        return out;
    }

    fragment float4 fragFunc(
        TexturePipelineRasterizerData in [[stage_in]],
        texture2d<float> texture [[texture(0)]],
        constant float& index [[buffer(29)]]
    ) {
        sampler imgSampler;
        float4 colorSample = texture.sample(imgSampler, in.texcoord);
        return colorSample * index;
    }
)END";

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    std::vector<float> a = {0.5};

    int width, height, nrChannels;
    unsigned char *data = stbi_load("yes.png", &width, &height, &nrChannels, 0);

    Render::Instance::Config renderCfg;
    renderCfg.title = "DankDemo!";
    renderCfg.size = {1920, 1080};
    renderCfg.resizable = true;
    renderCfg.imgui = true;
    renderCfg.debug = true;
    renderCfg.vsync = true;
    Render::Init(Render::Backend::Metal, renderCfg);

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = Render::Extras::FillScreenVertices();
    vertexCfg.indices = Render::Extras::FillScreenIndices();
    auto vertex = Render::Create(vertexCfg);

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Triangles;
    auto draw = Render::Create(drawVertexCfg);

    Render::Texture::Config imgCfg;
    imgCfg.size = {width, height};
    imgCfg.buffer = data;
    auto img = Render::Create(imgCfg);

    Render::Program::Config programCfg;
    programCfg.vertexSource = &shaders;
    programCfg.draws = {draw};
    programCfg.textures = {img};
    programCfg.uniforms = {
        {"alpha", &a},
    };
    auto program = Render::Create(programCfg);

    Render::Texture::Config textureCfg;
    textureCfg.size = {width, height};
    auto texture = Render::Create(textureCfg);

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    auto surface = Render::CreateAndBind(surfaceCfg);

    Render::Create();

    while (Render::KeepRunning()) {
        Render::Begin();

        // Create a window called "My First Tool", with a menu bar.
        ImGui::Begin("My First Tool", NULL, ImGuiWindowFlags_MenuBar);
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                ImGui::MenuItem("Open..", "Ctrl+O");
                ImGui::MenuItem("Save", "Ctrl+S");
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Plot some values
        const float my_values[] = { 0.2f, 0.1f, 1.0f, 0.5f, 0.9f, 2.2f };
        ImGui::PlotLines("Frame Times", my_values, IM_ARRAYSIZE(my_values));

        // Display contents in a scrolling region
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Important Stuff");
        ImGui::BeginChild("Scrolling");
        for (int n = 0; n < 50; n++)
            ImGui::Text("%04d: Some text", n);
        ImGui::EndChild();
        ImGui::End();

        ImGui::Begin("seY");
        ImGui::SliderFloat("Image Alpha", &a[0], 0.0, 1.0);
        auto [x, y] = ImGui::GetContentRegionAvail();
        auto [w, h] = surface->size({static_cast<int>(x), static_cast<int>(y)});
        ImGui::Image(texture->raw(), ImVec2(w, h));
        ImGui::End();

        Render::End();
    }

    Render::Destroy();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
