#ifndef RENDER_BASE_BACKEND_CYBERETHER_H
#define RENDER_BASE_BACKEND_CYBERETHER_H

#include <iostream>

#include "types.hpp"

namespace Render {

typedef struct {
    std::string title;
    int width;
    int height;
} Config;

class Render {
public:
    virtual ~Render() = default;
    virtual Backend getBackend() = 0;
    virtual Result init(Config) = 0;
};

} // namespace Render

#endif