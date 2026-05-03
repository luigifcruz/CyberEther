#pragma once

#include <string>

namespace Jetstream::Sakura {

struct NodeEditorPinRef {
    std::string nodeId;
    std::string pinId;
    bool isInput = false;
};

}  // namespace Jetstream::Sakura
