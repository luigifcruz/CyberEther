#ifndef JETSTREAM_BLOCK_SLICE_BASE_HH
#define JETSTREAM_BLOCK_SLICE_BASE_HH

#include <fcntl.h>
#include <regex>

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/tensor_modifier.hh"
#include "jetstream/modules/duplicate.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Slice : public Block {
 public:
    // Configuration

    struct Config {
        std::string slice = "[...]";
        bool contiguous = false;

        JST_SERDES(slice, contiguous);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "slice";
    }

    std::string name() const {
        return "Slice";
    }

    std::string summary() const {
        return "Creates a slice of a tensor.";
    }

    std::string description() const {
        return "Creates a slice of a tensor using NumPy-like slicing syntax. "
               "Optionally ensures the output data is contiguous in memory by copying "
               "the sliced data through the Duplicate module when the 'Contiguous' option is enabled. "
               "This guarantees optimal memory layout for subsequent operations that require contiguous data.";
    }

    // Constructor

    Result create() {
        const auto& sliceParser = [&](const std::string& slice, std::vector<Token>& tokens) {
            // Validate the overall structure of the slice string.

            if (slice.front() != '[' || slice.back() != ']') {
                JST_ERROR("Invalid slice syntax: Missing brackets.");
                return Result::ERROR;
            }

            // Return `[...]` if the slice is empty.

            std::string inner = slice.substr(1, slice.size() - 2);
            if (inner.empty()) {
                tokens.emplace_back("...");
                return Result::ERROR;
            }

            // Split the slice string into token strings.

            std::vector<std::string> elements;
            std::regex pattern(R"([^,\s\[\]]+)");
            auto words_begin = std::sregex_iterator(slice.begin(), slice.end(), pattern);
            auto words_end = std::sregex_iterator();

            for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                std::smatch match = *i;
                elements.push_back(match.str());
            }

            JST_TRACE("[SLICE] Found {} elements in slice string: {}", elements.size(), elements);

            // Parse the token strings into tokens.

            for (const auto& element : elements) {
                // Parse Ellipsis.

                if (element == "...") {
                    tokens.emplace_back("...");
                    JST_TRACE("[SLICE] Found ellipsis token.");
                    continue;
                }

                // Parse Colon.

                if (std::regex_match(element, std::regex(R"(^(\d+:\d+:\d+|\d+:\d+|:\d+|\d+:|:|::\d+)$)"))) {
                    std::regex pattern(R"((\d*):(\d*):?(\d*))");
                    std::smatch matches;

                    U64 a = 0, b = 0, c = 1;

                    if (std::regex_match(element, matches, pattern)) {
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
                        JST_TRACE("[SLICE] Found colon token: {}.", element);
                    }

                    continue;
                }

                // Parse Numbers.

                if (std::regex_match(element, std::regex(R"(\d+)"))) {
                    tokens.emplace_back(static_cast<U64>(std::stoull(element)));
                    JST_TRACE("[SLICE] Found number token: {}.", element);
                    continue;
                }

                JST_ERROR("Invalid slice syntax: Invalid token '{}'.", element);
                return Result::ERROR;
            }

            JST_TRACE("[SLICE] Parsed slice string {} to tokens {}.", slice, tokens);

            return Result::SUCCESS;
        };

        JST_CHECK(instance().addModule(
            modifier, "modifier", {
                .callback = [&](auto& mod) {
                    if (config.slice.empty()) {
                        return Result::SUCCESS;
                    }

                    std::vector<Token> tokens;
                    JST_CHECK(sliceParser(config.slice, tokens));
                    JST_CHECK(mod.slice(tokens));

                    return Result::SUCCESS;
                }
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        auto currentOutput = modifier->getOutputBuffer();

        if (config.contiguous) {
            JST_CHECK(instance().addModule(
                duplicate, "duplicate", {
                    .hostAccessible = true,
                }, {
                    .buffer = currentOutput,
                },
                locale()
            ));
            currentOutput = duplicate->getOutputBuffer();
        }

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, currentOutput));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (duplicate) {
            JST_CHECK(instance().eraseModule(duplicate->locale()));
        }

        if (modifier) {
            JST_CHECK(instance().eraseModule(modifier->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Slice");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##slice", &config.slice, ImGuiInputTextFlags_EnterReturnsTrue)) {
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Contiguous");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Checkbox("##contiguous", &config.contiguous)) {
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::TensorModifier<D, IT>> modifier;
    std::shared_ptr<Jetstream::Duplicate<D, IT>> duplicate;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Slice, !std::is_same<IT, void>::value &&
                         std::is_same<OT, void>::value &&
                         is_specialized<Jetstream::Duplicate<D, IT>>::value)

#endif
