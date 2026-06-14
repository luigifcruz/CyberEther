#ifndef JETSTREAM_DOMAINS_ML_INFER_MODULE_HH
#define JETSTREAM_DOMAINS_ML_INFER_MODULE_HH

#include <jetstream/module.hh>
#include <jetstream/types.hh>

namespace Jetstream::Modules {

struct Infer : public Module::Config {
    std::string modelPath = "";
    std::string inputName = "modelInput";
    std::string outputName = "output";
    U64 batchSize = 1;
    std::string executionProvider = "cpu";

    JST_MODULE_TYPE(infer);
    JST_MODULE_PARAMS(modelPath, inputName, outputName, batchSize, executionProvider);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_ML_INFER_MODULE_HH
