#ifndef SPECTRUM_CYBERETHER_H
#define SPECTRUM_CYBERETHER_H

#include <memory>

#include "render.hpp"

namespace Spectrum {

typedef struct {
    std::shared_ptr<Render::API> render;
} Config;

class View {
public:
    Result init(Config);
};

inline std::shared_ptr<View> instantiate() {
    return std::make_shared<View>();
}

} // namespace Spectrum

#endif