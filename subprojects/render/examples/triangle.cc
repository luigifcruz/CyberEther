#include "render/base.hpp"

Result render() {
    Render::Instance::Config renderCfg;
    renderCfg.size = {3130, 1140};
    renderCfg.resizable = true;
    renderCfg.imgui = true;
    renderCfg.vsync = true;
    renderCfg.title = "Demo";
    auto render = Render::Instantiate(Render::API::METAL, renderCfg);

    render->create();

    return Result::SUCCESS;
}

int main() {
    if (render() != Result::SUCCESS) {
        return 1;
        std::cout << "Error!" << std::endl;
    }

    return 0;

}
