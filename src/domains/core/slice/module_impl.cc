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
    // Return empty if the slice content is empty.
    std::string inner = sliceStr.substr(1, sliceStr.size() - 2);
    if (inner.empty()) {
        tokens.emplace_back("...");
        return Result::SUCCESS;
    }

    // Split the slice string into token strings.
    std::vector<std::string> elements;
    std::regex pattern(R"([^,\s\[\]]+)");
    auto words_begin = std::sregex_iterator(sliceStr.begin(), sliceStr.end(), pattern);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        elements.push_back(match.str());
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
        if (std::regex_match(element,
                std::regex(R"(^(\d+:\d+:\d+|\d+:\d+|:\d+|\d+:|:|::\d+)$)"))) {
            std::regex colonPattern(R"((\d*):(\d*):?(\d*))");
            std::smatch matches;

            U64 a = 0, b = 0, c = 1;

            if (std::regex_match(element, matches, colonPattern)) {
                if (matches.size() > 1 && matches[1].matched && !matches[1].str().empty()) {
                    a = std::stoull(matches[1].str());
                }
                if (matches.size() > 2 && matches[2].matched && !matches[2].str().empty()) {
                    b = std::stoull(matches[2].str());
                }
                if (matches.size() > 3 && matches[3].matched && !matches[3].str().empty()) {
                    c = std::stoull(matches[3].str());
                }

                tokens.emplace_back(a, b, c);
                JST_TRACE("[MODULE_SLICE] Found colon token: {}.", element);
            }

            continue;
        }

        // Parse Numbers.
        if (std::regex_match(element, std::regex(R"(\d+)"))) {
            tokens.emplace_back(static_cast<U64>(std::stoull(element)));
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
