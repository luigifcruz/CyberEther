#include <exception>

#include "module_impl.hh"

namespace Jetstream::Modules {

Result SliceImpl::validate() {
    const auto& config = *candidate();

    if (config.slice.empty()) {
        JST_ERROR("[MODULE_SLICE] Slice string cannot be empty.");
        return Result::ERROR;
    }

    if (config.slice.front() != '[' || config.slice.back() != ']') {
        JST_ERROR("[MODULE_SLICE] Invalid slice syntax: Missing brackets.");
        return Result::ERROR;
    }

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

    if (!slice.empty() && slice != "[...]") {
        std::vector<Token> tokens;
        JST_CHECK(parseSliceString(slice, tokens));
        JST_CHECK(output.slice(tokens));
    }

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

Result SliceImpl::parseSliceString(const std::string& sliceStr,
                                   std::vector<Token>& tokens) {
    std::string inner = sliceStr.substr(1, sliceStr.size() - 2);
    constexpr auto whitespace = " \t\n\r\f\v";
    const auto contentStart = inner.find_first_not_of(whitespace);
    if (contentStart == std::string::npos) {
        tokens.emplace_back("...");
        return Result::SUCCESS;
    }
    inner = inner.substr(contentStart, inner.find_last_not_of(whitespace) - contentStart + 1);

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
        element = element.substr(tokenStart, element.find_last_not_of(whitespace) - tokenStart + 1);
        elements.push_back(std::move(element));

        if (comma == std::string::npos) {
            break;
        }
        elementStart = comma + 1;
    }

    JST_TRACE("[MODULE_SLICE] Found {} elements in slice string: {}", elements.size(), elements);

    // Parse the token strings into tokens.
    for (const auto& element : elements) {
        // Parse Ellipsis.
        if (element == "...") {
            tokens.emplace_back("...");
            JST_TRACE("[MODULE_SLICE] Found ellipsis token.");
            continue;
        }

        // Parse Colon notation (start:stop:step).
        if (std::regex_match(element, std::regex(R"(^(\d+:\d+:\d+|:\d+:\d+|\d+::\d+|\d+:\d+|:\d+|\d+:|:|::\d+)$)"))) {
            std::regex colonPattern(R"((\d*):(\d*):?(\d*))");
            std::smatch matches;

            U64 a = 0, b = 0, c = 1;
            bool hasEnd = false;

            try {
                if (std::regex_match(element, matches, colonPattern)) {
                    if (matches.size() > 1 && matches[1].matched && !matches[1].str().empty()) {
                        a = std::stoull(matches[1].str());
                    }
                    if (matches.size() > 2 && matches[2].matched && !matches[2].str().empty()) {
                        b = std::stoull(matches[2].str());
                        hasEnd = true;
                    }
                    if (matches.size() > 3 && matches[3].matched && !matches[3].str().empty()) {
                        c = std::stoull(matches[3].str());
                    }

                    tokens.emplace_back(a, b, c, hasEnd);
                    JST_TRACE("[MODULE_SLICE] Found colon token: {}.", element);
                }
            } catch (const std::exception&) {
                JST_ERROR("[MODULE_SLICE] Invalid numeric value in token '{}'.", element);
                return Result::ERROR;
            }

            continue;
        }

        // Parse Numbers.
        if (std::regex_match(element, std::regex(R"(\d+)"))) {
            try {
                tokens.emplace_back(static_cast<U64>(std::stoull(element)));
            } catch (const std::exception&) {
                JST_ERROR("[MODULE_SLICE] Invalid numeric value in token '{}'.", element);
                return Result::ERROR;
            }
            JST_TRACE("[MODULE_SLICE] Found number token: {}.", element);
            continue;
        }

        JST_ERROR("[MODULE_SLICE] Invalid slice syntax: Invalid token '{}'.", element);
        return Result::ERROR;
    }

    JST_TRACE("[MODULE_SLICE] Parsed slice string {} to tokens {}.", sliceStr, tokens);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
