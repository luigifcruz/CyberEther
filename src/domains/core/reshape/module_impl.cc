#include "module_impl.hh"

namespace Jetstream::Modules {

Result ReshapeImpl::validate() {
    const auto& config = *candidate();

    if (config.shape.empty()) {
        JST_ERROR("[MODULE_RESHAPE] Shape string cannot be empty.");
        return Result::ERROR;
    }

    if (config.shape.front() != '[' || config.shape.back() != ']') {
        JST_ERROR("[MODULE_RESHAPE] Invalid shape syntax: Missing brackets.");
        return Result::ERROR;
    }

    Shape tempShape;
    JST_CHECK(parseShapeString(config.shape, tempShape));

    if (tempShape.empty()) {
        JST_ERROR("[MODULE_RESHAPE] Shape must have at least one dimension.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ReshapeImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result ReshapeImpl::create() {
    JST_CHECK(parseShapeString(shape, parsedShape));

    const Tensor& inputTensor = inputs().at("buffer").tensor;

    if (!inputTensor.contiguous()) {
        JST_ERROR("[MODULE_RESHAPE] Cannot reshape non-contiguous tensor. "
                  "Use the contiguous option or duplicate the tensor first.");
        return Result::ERROR;
    }

    // Calculate total elements in input.
    U64 inputSize = 1;
    for (const auto dim : inputTensor.shape()) {
        inputSize *= dim;
    }

    // Calculate total elements in target shape.
    U64 targetSize = 1;
    for (const auto dim : parsedShape) {
        targetSize *= dim;
    }

    if (inputSize != targetSize) {
        JST_ERROR("[MODULE_RESHAPE] Cannot reshape tensor with {} elements to shape with {} elements.",
                  inputSize, targetSize);
        return Result::ERROR;
    }

    input = inputTensor;
    output = input;

    JST_CHECK(output.reshape(parsedShape));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"] = {name(), "buffer", output};

    return Result::SUCCESS;
}

Result ReshapeImpl::parseShapeString(const std::string& shapeStr, Shape& result) {
    result.clear();

    // Return empty if the shape content is empty (just "[]").
    std::string inner = shapeStr.substr(1, shapeStr.size() - 2);
    if (inner.empty()) {
        return Result::SUCCESS;
    }

    // Extract all numbers from the shape string.
    std::regex pattern(R"(\d+)");
    auto numbers_begin = std::sregex_iterator(inner.begin(), inner.end(), pattern);
    auto numbers_end = std::sregex_iterator();

    for (std::sregex_iterator i = numbers_begin; i != numbers_end; ++i) {
        std::smatch match = *i;
        U64 dim = std::stoull(match.str());
        if (dim == 0) {
            JST_ERROR("[MODULE_RESHAPE] Shape dimensions cannot be zero.");
            return Result::ERROR;
        }
        result.push_back(dim);
    }

    JST_TRACE("[MODULE_RESHAPE] Parsed shape string '{}' to {}.", shapeStr, result);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
