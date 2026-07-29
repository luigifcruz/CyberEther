#include "module_impl.hh"

#include <charconv>
#include <cctype>
#include <system_error>

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

Result ParseShapeString(const std::string& shapeStr, Shape& result) {
    if (shapeStr.size() < 2 || shapeStr.front() != '[' || shapeStr.back() != ']') {
        JST_ERROR("[MODULE_RESHAPE] Shape must use bracket notation.");
        return Result::ERROR;
    }

    Shape parsed;
    std::size_t position = 1;
    const std::size_t closingBracket = shapeStr.size() - 1;
    const auto skipWhitespace = [&]() {
        while (position < closingBracket &&
               std::isspace(static_cast<unsigned char>(shapeStr[position]))) {
            ++position;
        }
    };

    skipWhitespace();
    if (position == closingBracket) {
        JST_ERROR("[MODULE_RESHAPE] Shape must have at least one dimension.");
        return Result::ERROR;
    }

    while (position < closingBracket) {
        const std::size_t numberBegin = position;
        while (position < closingBracket && shapeStr[position] >= '0' &&
               shapeStr[position] <= '9') {
            ++position;
        }

        if (numberBegin == position) {
            JST_ERROR("[MODULE_RESHAPE] Invalid shape syntax '{}'.", shapeStr);
            return Result::ERROR;
        }

        U64 dimension = 0;
        const char* begin = shapeStr.data() + numberBegin;
        const char* end = shapeStr.data() + position;
        const auto conversion = std::from_chars(begin, end, dimension);
        if (conversion.ec == std::errc::result_out_of_range) {
            JST_ERROR("[MODULE_RESHAPE] Shape dimension exceeds the supported "
                      "numeric range.");
            return Result::ERROR;
        }
        if (conversion.ec != std::errc{} || conversion.ptr != end) {
            JST_ERROR("[MODULE_RESHAPE] Invalid shape dimension.");
            return Result::ERROR;
        }

        if (dimension == 0) {
            JST_ERROR("[MODULE_RESHAPE] Shape dimensions cannot be zero.");
            return Result::ERROR;
        }
        parsed.push_back(dimension);

        skipWhitespace();
        if (position == closingBracket) {
            break;
        }
        if (shapeStr[position] != ',') {
            JST_ERROR("[MODULE_RESHAPE] Invalid shape syntax '{}'.", shapeStr);
            return Result::ERROR;
        }

        ++position;
        skipWhitespace();
        if (position == closingBracket) {
            JST_ERROR("[MODULE_RESHAPE] Invalid shape syntax '{}'.", shapeStr);
            return Result::ERROR;
        }
    }

    result = parsed;
    JST_TRACE("[MODULE_RESHAPE] Parsed shape string '{}' to {}.", shapeStr, result);

    return Result::SUCCESS;
}

}  // namespace

Result ReshapeImpl::validate() {
    const auto& config = *candidate();

    Shape candidateShape;
    JST_CHECK(ParseShapeString(config.shape, candidateShape));

    U64 targetSize = 1;
    for (const U64 dimension : candidateShape) {
        if (!detail::CheckedMultiply(targetSize, dimension, targetSize)) {
            JST_ERROR("[MODULE_RESHAPE] Shape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    if (inputs().contains("buffer")) {
        const Tensor& inputTensor = inputs().at("buffer").tensor;
        if (inputTensor.validShape() && inputTensor.size() > 0) {
            if (!inputTensor.contiguous()) {
                JST_ERROR("[MODULE_RESHAPE] Cannot reshape non-contiguous tensor. "
                          "Use the contiguous option or duplicate the tensor first.");
                return Result::ERROR;
            }

            if (inputTensor.size() != targetSize) {
                JST_ERROR("[MODULE_RESHAPE] Cannot reshape tensor with {} elements "
                          "to shape with {} elements.",
                          inputTensor.size(), targetSize);
                return Result::ERROR;
            }
        }
    }

    parsedShape = candidateShape;
    return Result::SUCCESS;
}

Result ReshapeImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result ReshapeImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;
    output = input.clone();

    JST_CHECK(output.reshape(parsedShape));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
