#ifndef JETSTREAM_DOMAINS_CORE_PYTHON_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_PYTHON_MODULE_HH

#include <string>
#include <vector>

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Python : public Module::Config {
    struct TensorSpec {
        std::string shape = "[1]";
        std::string dtype = "F32";
        std::string device = "cpu";

        bool operator==(const TensorSpec&) const = default;

        JST_SERDES(shape, dtype, device);
    };

    std::string code = R"PY(def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0]
    )PY";
    U64 inputCount = 1;
    U64 outputCount = 1;
    std::vector<TensorSpec> outputTensorSpecs;

    JST_MODULE_TYPE(python);
    JST_MODULE_PARAMS(code, inputCount, outputCount,
                      outputTensorSpecs);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_PYTHON_MODULE_HH
