#ifndef JETSTREAM_TESTS_ONNX_MODEL_HELPERS_HH
#define JETSTREAM_TESTS_ONNX_MODEL_HELPERS_HH

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <string>

namespace Jetstream::Tests {

inline std::filesystem::path CreateTempOnnxModelPath(const std::string& stem) {
    const auto base = std::filesystem::temp_directory_path() / "cyberether-onnx-tests";
    std::filesystem::create_directories(base);

    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return base / (stem + "-" + unique + ".onnx");
}

inline std::filesystem::path CreateIdentityOnnxModelWithDtypes(const std::string& inputTensorProtoType,
                                                               const std::string& outputTensorProtoType,
                                                               const std::string& stem);

inline std::filesystem::path CreateIdentityOnnxModel(const std::string& tensorProtoType,
                                                     const std::string& stem) {
    return CreateIdentityOnnxModelWithDtypes(tensorProtoType, tensorProtoType, stem);
}

inline std::filesystem::path CreateIdentityOnnxModelWithDtypes(const std::string& inputTensorProtoType,
                                                               const std::string& outputTensorProtoType,
                                                               const std::string& stem) {
    const auto modelPath = CreateTempOnnxModelPath(stem);
    const auto scriptPath = modelPath.parent_path() / (stem + ".py");

    {
        std::ofstream script(scriptPath);
        script
            << "import sys\n"
            << "import onnx\n"
            << "from onnx import TensorProto, helper\n"
            << "output_path = sys.argv[1]\n"
            << "input_tensor_proto_name = sys.argv[2]\n"
            << "output_tensor_proto_name = sys.argv[3]\n"
            << "input_tensor_type = getattr(TensorProto, input_tensor_proto_name)\n"
            << "output_tensor_type = getattr(TensorProto, output_tensor_proto_name)\n"
            << "graph = helper.make_graph(\n"
            << "    [helper.make_node('Identity', ['input_0'], ['output_0'])],\n"
            << "    'identity',\n"
            << "    [helper.make_tensor_value_info('input_0', input_tensor_type, [4])],\n"
            << "    [helper.make_tensor_value_info('output_0', output_tensor_type, [4])],\n"
            << ")\n"
            << "model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])\n"
            << "model.ir_version = 11\n"
            << "onnx.checker.check_model(model)\n"
            << "onnx.save(model, output_path)\n";
    }

    const auto cmd = "python3 '" + scriptPath.string() + "' '" + modelPath.string() + "' '" +
                     inputTensorProtoType + "' '" + outputTensorProtoType + "'";
    const auto status = std::system(cmd.c_str());
    if (status != 0) {
        throw std::runtime_error("failed to generate ONNX identity model");
    }

    std::error_code ec;
    std::filesystem::remove(scriptPath, ec);

    return modelPath;
}

}  // namespace Jetstream::Tests

#endif  // JETSTREAM_TESTS_ONNX_MODEL_HELPERS_HH
