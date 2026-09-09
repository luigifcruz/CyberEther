#ifndef JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_HH

#include <string>

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Frame : public Module::Config {
    std::string fit = "contain";
    std::string colormap = "grayscale";
    bool autoRange = true;
    bool smooth = false;
    std::string xLabel = "X (px)";
    std::string yLabel = "Y (px)";

    JST_MODULE_TYPE(frame);
    JST_MODULE_PARAMS(fit, colormap, autoRange, smooth, xLabel, yLabel);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_HH
