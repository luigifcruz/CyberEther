#include "jetstream/modules/lineplot.hh"
#include "shaders/lineplot_shaders.hh"

namespace Jetstream {

template<Device D, typename T>
Lineplot<D, T>::Lineplot(const Config& config,
                         const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Lineplot module.");
    JST_CHECK_THROW(Module::initInput(input.buffer));
}

template<Device D, typename T>
Result Lineplot<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Lineplot compute core using CPU backend.");
    
    {
        // Generate Grid coordinates.
        const U64 num_cols = config.numberOfVerticalLines;
        const U64 num_rows = config.numberOfHorizontalLines;
        
        grid = Vector<D, F32, 3>({num_cols + num_rows, 2, 3});

        const F32 x_step  = +2.0f / (num_cols - 1);
        const F32 y_step  = +2.0f / (num_rows - 1);
        const F32 x_start = -1.0f;
        const F32 y_start = -1.0f;
        const F32 x_end   = +1.0f;
        const F32 y_end   = +1.0f;

        for (U64 row = 0; row < num_rows; row++) {
            const F32 y = y_start + row * y_step;
            
            grid[{row, 0, 0}] = x_start;
            grid[{row, 0, 1}] = y;

            grid[{row, 1, 0}] = x_end;
            grid[{row, 1, 1}] = y;
        }

        for (U64 col = 0; col < num_cols; col++) {
            const F32 x = x_start + col * x_step;
            
            grid[{col + num_rows, 0, 0}] = x;
            grid[{col + num_rows, 0, 1}] = y_start;

            grid[{col + num_rows, 1, 0}] = x;
            grid[{col + num_rows, 1, 1}] = y_end;
        }
    }

    {
        // Generate Plot coordinates.
        const U64 num_cols = input.buffer.shape()[1];

        plot = Vector<D, F32, 2>({num_cols, 3});

        for (U64 j = 0; j < num_cols; j++) {
            plot[{j, 0}] = j * 2.0f / (num_cols - 1) - 1.0f;
        }
    }

    JST_CHECK(underlyingCreateCompute(meta));
    
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::createPresent(Render::Window& window) {
    Render::Buffer::Config gridVerticesConf;
    gridVerticesConf.buffer = grid.data();
    gridVerticesConf.elementByteSize = sizeof(grid[0]);
    gridVerticesConf.size = grid.size();
    gridVerticesConf.target = Render::Buffer::Target::VERTEX;
    gridVerticesConf.enableZeroCopy = true;
    JST_CHECK(window.build(gridVerticesBuffer, gridVerticesConf));

    Render::Vertex::Config gridVertexCfg;
    gridVertexCfg.buffers = {
        {gridVerticesBuffer, 3},
    };
    JST_CHECK(window.build(gridVertex, gridVertexCfg));

    Render::Draw::Config drawGridVertexCfg;
    drawGridVertexCfg.buffer = gridVertex;
    drawGridVertexCfg.mode = Render::Draw::Mode::LINES;
    JST_CHECK(window.build(drawGridVertex, drawGridVertexCfg));

    Render::Buffer::Config lineVerticesConf;
    lineVerticesConf.buffer = plot.data();
    lineVerticesConf.elementByteSize = sizeof(plot[0]);
    lineVerticesConf.size = plot.size();
    lineVerticesConf.target = Render::Buffer::Target::VERTEX;
    lineVerticesConf.enableZeroCopy = true;
    JST_CHECK(window.build(lineVerticesBuffer, lineVerticesConf));

    Render::Vertex::Config lineVertexCfg;
    lineVertexCfg.buffers = {
        {lineVerticesBuffer, 3},
    };
    JST_CHECK(window.build(lineVertex, lineVertexCfg));

    Render::Draw::Config drawLineVertexCfg;
    drawLineVertexCfg.buffer = lineVertex;
    drawLineVertexCfg.mode = Render::Draw::Mode::LINE_STRIP;
    JST_CHECK(window.build(drawLineVertex, drawLineVertexCfg));

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
    JST_CHECK(window.build(lutTexture, lutTextureCfg));

    Render::Program::Config gridProgramCfg;
    gridProgramCfg.shaders = ShadersPackage["grid"];
    gridProgramCfg.draw = drawGridVertex;
    JST_CHECK(window.build(gridProgram, gridProgramCfg));

    Render::Program::Config signalProgramCfg;
    signalProgramCfg.shaders = ShadersPackage["signal"];
    signalProgramCfg.draw = drawLineVertex;
    signalProgramCfg.textures = {lutTexture};
    JST_CHECK(window.build(signalProgram, signalProgramCfg));

    Render::Texture::Config textureCfg;
    textureCfg.size = config.viewSize;
    JST_CHECK(window.build(texture, textureCfg));

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {gridProgram, signalProgram};
    JST_CHECK(window.build(surface, surfaceCfg));
    JST_CHECK(window.bind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
void Lineplot<D, T>::summary() const {
    JST_INFO("     Size: [{}, {}]", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
Result Lineplot<D, T>::present(Render::Window&) {
    lineVerticesBuffer->update();
    return Result::SUCCESS;
}

template<Device D, typename T>
const Render::Size2D<U64>& Lineplot<D, T>::viewSize(const Render::Size2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        this->config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Lineplot<D, T>::getTexture() {
    return *texture;
};

template<Device D, typename T>
Result Lineplot<D, T>::Factory(std::unordered_map<std::string, std::any>& configMap,
                               std::unordered_map<std::string, std::any>& inputMap,
                               std::unordered_map<std::string, std::any>&,
                               std::shared_ptr<Lineplot<D, T>>& module) {
    using Module = Lineplot<D, T>;

    Module::Config config{};

    JST_CHECK(Module::BindVariable(configMap, "numberOfVerticalLines", config.numberOfVerticalLines));
    JST_CHECK(Module::BindVariable(configMap, "numberOfHorizontalLines", config.numberOfHorizontalLines));
    JST_CHECK(Module::BindVariable(configMap, "viewSize", config.viewSize));

    Module::Input input{};

    JST_CHECK(Module::BindVariable(inputMap, "buffer", input.buffer));

    module = std::make_shared<Module>(config, input);

    return Result::SUCCESS;
}

}  // namespace Jetstream
