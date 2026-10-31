#include "support/base.hh"

namespace Jetstream::Agent {

namespace {

using Detail::Diagnostics;

constexpr const char* kCatalogHost = "https://models.dev";
constexpr const char* kCatalogContext = "models.dev model catalog";

void ParseReasoningEfforts(const nlohmann::json& model, auto& entry) {
    const auto options = model.find("reasoning_options");
    if (options == model.end() || !options->is_array()) {
        return;
    }

    for (const auto& option : *options) {
        const auto values = option.find("values");
        const bool isEffort = option.value("type", "") == "effort";
        if (!isEffort || values == option.end() || !values->is_array()) {
            continue;
        }

        for (const auto& value : *values) {
            if (value.is_string()) {
                entry.reasoningEfforts.push_back(value.get<std::string>());
            }
        }
    }
}

void ParseLimits(const nlohmann::json& model, auto& entry) {
    const auto limit = model.find("limit");
    if (limit == model.end() || !limit->is_object()) {
        return;
    }

    const auto context = limit->find("context");
    if (context != limit->end() && context->is_number_unsigned()) {
        entry.contextLength = context->get<U64>();
    }

    const auto output = limit->find("output");
    if (output != limit->end() && output->is_number_unsigned()) {
        entry.outputLength = output->get<U64>();
    }
}

void ParseModel(const nlohmann::json& model, auto& entry) {
    const auto name = model.find("name");
    if (name != model.end() && name->is_string()) {
        entry.name = name->get<std::string>();
    }

    ParseReasoningEfforts(model, entry);
    ParseLimits(model, entry);
}

}  // namespace

Result Provider::Catalog::fetch(Provider::Error* error, const Cancellation& cancelled) {
    entries.clear();
    if (error) {
        *error = {};
    }

    try {
        Platform::Http::Client::Request http;
        http.host = kCatalogHost;
        http.path = "/api.json";
        http.timeoutSeconds = 15;
        http.cancelled = cancelled;

        nlohmann::json body;
        if (auto failure = Diagnostics::Response(Platform::Http::Client::Send(http), body)) {
            return Diagnostics::Report(std::move(failure), error, kCatalogContext, {});
        }

        if (!body.is_object()) {
            return Diagnostics::Report({Provider::Error::Kind::Protocol, 0, "The model catalog returned malformed JSON."},
                                       error, kCatalogContext, {});
        }

        for (const auto& profile : Profile::All()) {
            if (profile.catalogId.empty()) {
                continue;
            }

            const auto provider = body.find(profile.catalogId);
            if (provider == body.end() || !provider->is_object()) {
                continue;
            }

            const auto models = provider->find("models");
            if (models == provider->end() || !models->is_object()) {
                continue;
            }

            for (const auto& [id, model] : models->items()) {
                if (!model.is_object()) {
                    continue;
                }

                auto& entry = entries[profile.catalogId + '\n' + id];
                entry.id = id;
                ParseModel(model, entry);
            }
        }

        return Result::SUCCESS;
    } catch (const std::exception& exception) {
        return Diagnostics::Report({Provider::Error::Kind::Protocol, 0, exception.what()}, error, kCatalogContext, {});
    }
}

std::optional<Provider::Model> Provider::Catalog::find(Kind kind, const std::string& id) const {
    const auto& catalogId = Profile::Find(kind).catalogId;
    if (catalogId.empty()) {
        return std::nullopt;
    }

    const auto found = entries.find(catalogId + '\n' + id);
    if (found == entries.end()) {
        return std::nullopt;
    }
    return found->second;
}

}  // namespace Jetstream::Agent
