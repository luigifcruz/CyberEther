#include "module_impl.hh"

#include <algorithm>
#include <charconv>
#include <string_view>
#include <system_error>
#include <utility>

namespace Jetstream::Modules {

namespace {

Result ParseSliceString(const std::string& sliceStr,
                        std::vector<Token>& tokens) {
    if (sliceStr.empty()) {
        JST_ERROR("[MODULE_SLICE] Slice string cannot be empty.");
        return Result::ERROR;
    }
    if (sliceStr.front() != '[' || sliceStr.back() != ']') {
        JST_ERROR("[MODULE_SLICE] Invalid slice syntax: Missing brackets.");
        return Result::ERROR;
    }

    std::string inner = sliceStr.substr(1, sliceStr.size() - 2);
    constexpr auto whitespace = " \t\n\r\f\v";
    const auto contentStart = inner.find_first_not_of(whitespace);
    if (contentStart == std::string::npos) {
        tokens = {Token("...")};
        return Result::SUCCESS;
    }
    inner = inner.substr(contentStart,
                         inner.find_last_not_of(whitespace) - contentStart + 1);

    std::vector<std::string> elements;
    std::size_t elementStart = 0;
    while (elementStart <= inner.size()) {
        const auto comma = inner.find(',', elementStart);
        std::string element = inner.substr(elementStart, comma - elementStart);
        const auto tokenStart = element.find_first_not_of(whitespace);
        if (tokenStart == std::string::npos) {
            JST_ERROR("[MODULE_SLICE] Invalid slice syntax: Empty token.");
            return Result::ERROR;
        }
        element = element.substr(tokenStart,
                                 element.find_last_not_of(whitespace) - tokenStart + 1);
        elements.push_back(std::move(element));

        if (comma == std::string::npos) {
            break;
        }
        elementStart = comma + 1;
    }

    JST_TRACE("[MODULE_SLICE] Found {} elements in slice string: {}",
              elements.size(), elements);

    std::vector<Token> parsedTokens;
    const auto isUnsignedInteger = [](const std::string_view value) {
        return !value.empty() && std::all_of(value.begin(), value.end(), [](const char ch) {
            return ch >= '0' && ch <= '9';
        });
    };
    const auto parseUnsigned = [&](const std::string_view value, U64& result,
                                   const std::string& element) {
        const auto conversion = std::from_chars(value.data(), value.data() + value.size(),
                                                result);
        if (conversion.ec == std::errc::result_out_of_range) {
            JST_ERROR("[MODULE_SLICE] Invalid numeric value in token '{}'.", element);
            return Result::ERROR;
        }
        if (conversion.ec != std::errc{} ||
            conversion.ptr != value.data() + value.size()) {
            JST_ERROR("[MODULE_SLICE] Invalid slice syntax: Invalid token '{}'.", element);
            return Result::ERROR;
        }
        return Result::SUCCESS;
    };

    for (const auto& element : elements) {
        if (element == "...") {
            parsedTokens.emplace_back("...");
            JST_TRACE("[MODULE_SLICE] Found ellipsis token.");
            continue;
        }

        const auto firstColon = element.find(':');
        if (firstColon != std::string::npos) {
            const auto secondColon = element.find(':', firstColon + 1);
            if ((secondColon != std::string::npos &&
                 element.find(':', secondColon + 1) != std::string::npos) ||
                (secondColon != std::string::npos && secondColon + 1 == element.size())) {
                JST_ERROR("[MODULE_SLICE] Invalid slice syntax: Invalid token '{}'.",
                          element);
                return Result::ERROR;
            }

            const std::string_view elementView(element);
            const auto startText = elementView.substr(0, firstColon);
            const auto endText = secondColon == std::string::npos
                                     ? elementView.substr(firstColon + 1)
                                     : elementView.substr(firstColon + 1,
                                                          secondColon - firstColon - 1);
            const auto stepText = secondColon == std::string::npos
                                      ? std::string_view{}
                                      : elementView.substr(secondColon + 1);
            if ((!startText.empty() && !isUnsignedInteger(startText)) ||
                (!endText.empty() && !isUnsignedInteger(endText)) ||
                (secondColon != std::string::npos && !isUnsignedInteger(stepText))) {
                JST_ERROR("[MODULE_SLICE] Invalid slice syntax: Invalid token '{}'.",
                          element);
                return Result::ERROR;
            }

            U64 start = 0;
            U64 end = 0;
            U64 step = 1;
            if (!startText.empty()) {
                JST_CHECK(parseUnsigned(startText, start, element));
            }
            if (!endText.empty()) {
                JST_CHECK(parseUnsigned(endText, end, element));
            }
            if (!stepText.empty()) {
                JST_CHECK(parseUnsigned(stepText, step, element));
            }

            parsedTokens.emplace_back(start, end, step, !endText.empty());
            JST_TRACE("[MODULE_SLICE] Found colon token: {}.", element);
            continue;
        }

        if (isUnsignedInteger(element)) {
            U64 index = 0;
            JST_CHECK(parseUnsigned(element, index, element));
            parsedTokens.emplace_back(index);
            JST_TRACE("[MODULE_SLICE] Found number token: {}.", element);
            continue;
        }

        JST_ERROR("[MODULE_SLICE] Invalid slice syntax: Invalid token '{}'.", element);
        return Result::ERROR;
    }

    const auto ellipsisCount = std::count_if(
        parsedTokens.begin(), parsedTokens.end(), [](const auto& token) {
            return token.getType() == Token::Type::Ellipsis;
        });
    if (ellipsisCount > 1) {
        JST_ERROR("[MODULE_SLICE] Ellipsis can only appear once in a slice.");
        return Result::ERROR;
    }
    for (const auto& token : parsedTokens) {
        if ((token.getType() == Token::Type::Colon ||
             token.getType() == Token::Type::ColonZeroEnd) &&
            token.getC() == 0) {
            JST_ERROR("[MODULE_SLICE] Slice step cannot be zero.");
            return Result::ERROR;
        }
    }

    tokens = std::move(parsedTokens);
    JST_TRACE("[MODULE_SLICE] Parsed slice string {} to tokens {}.", sliceStr, tokens);

    return Result::SUCCESS;
}

}  // namespace

Result SliceImpl::validate() {
    const auto& config = *candidate();

    std::vector<Token> candidatePlan;
    JST_CHECK(ParseSliceString(config.slice, candidatePlan));

    Tensor::SlicePlan candidateSlicePlan;
    if (inputs().contains("buffer")) {
        const Tensor& inputTensor = inputs().at("buffer").tensor;
        if (inputTensor.validShape() && inputTensor.size() > 0) {
            JST_CHECK(inputTensor.planSlice(candidatePlan, candidateSlicePlan));
        }
    }

    slicePlan = std::move(candidateSlicePlan);
    return Result::SUCCESS;
}

Result SliceImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result SliceImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;
    output = input;

    JST_CHECK(output.applySlicePlan(slicePlan));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
