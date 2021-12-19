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
        texture2d<float> texture [[texture(0)]]
    ) {
        sampler imgSampler;
        float4 colorSample = texture.sample(imgSampler, in.texcoord);
        return colorSample;
    }
)END";

int width, height, nrChannels;

std::shared_ptr<Render::Instance> render;
std::shared_ptr<Render::Texture> texture;
std::shared_ptr<Render::Texture> img;
std::shared_ptr<Render::Surface> surface;
std::shared_ptr<Render::Program> program;
std::shared_ptr<Render::Vertex> vertex;
std::shared_ptr<Render::Draw> draw;

void render_loop() {
    render->begin();

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

    ImGui::Begin("Waterfall");
    auto [x, y] = ImGui::GetContentRegionAvail();
    auto [width, height] = surface->size({(int)x, (int)y});
    ImGui::Image(texture->raw(), ImVec2(width, height));
    ImGui::End();

    render->end();
}

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    unsigned char *data = stbi_load("yes.png", &width, &height, &nrChannels, 0);

    Render::Instance::Config renderCfg;
    renderCfg.size = {1920, 1080};
    renderCfg.resizable = true;
    renderCfg.imgui = true;
    renderCfg.debug = true;
    renderCfg.vsync = true;
    renderCfg.title = "DankDemo!";
    render = Render::Instantiate(Render::API::METAL, renderCfg);

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = Render::Extras::FillScreenVertices();
    vertexCfg.indices = Render::Extras::FillScreenIndices();
    vertex = render->create(vertexCfg);

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Triangles;
    draw = render->create(drawVertexCfg);

    Render::Texture::Config imgCfg;
    imgCfg.size = {width, height};
    imgCfg.buffer = data;
    img = render->create(imgCfg);

    Render::Program::Config programCfg;
    programCfg.vertexSource = &shaders;
    programCfg.draws = {draw};
    programCfg.textures = {img};
    program = render->create(programCfg);

    Render::Texture::Config textureCfg;
    textureCfg.size = {width, height};
    texture = render->create(textureCfg);

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    surface = render->createAndBind(surfaceCfg);

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
