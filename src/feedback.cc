#include "feedback.hh"

#include "jetstream/logger.hh"

#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#if defined(JST_OS_BROWSER)
#include <emscripten/fetch.h>
#elif defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include <httplib.h>
#endif

namespace Jetstream {

namespace {

std::string BuildFeedbackBody(const std::string& text) {
    nlohmann::json json;
    json["feedback"] = text;
    return json.dump();
}

}  // namespace

struct Feedback::Impl {
    mutable std::mutex mutex;
    Snapshot state;
    bool active = true;
    U64 generation = 0;

#if !defined(JST_OS_BROWSER) && defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
    std::thread worker;
#endif

    void setSuccess(U64 gen, const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex);
        if (gen != generation || !active) {
            return;
        }
        state.status = Snapshot::Status::Success;
        state.id = id;
        state.message = "Thank you! Your feedback has been submitted.";
    }

    void setError(U64 gen, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex);
        if (gen != generation || !active) {
            return;
        }
        state.status = Snapshot::Status::Error;
        state.message = message;
    }

    void setSubmitting() {
        std::lock_guard<std::mutex> lock(mutex);
        ++generation;
        state.status = Snapshot::Status::Submitting;
        state.message.clear();
        state.id.clear();
    }

    U64 currentGeneration() const {
        std::lock_guard<std::mutex> lock(mutex);
        return generation;
    }
};

#if defined(JST_OS_BROWSER)
struct FetchContext {
    std::shared_ptr<Feedback::Impl> impl;
    U64 generation;
    std::string body;
};
#endif

Feedback::Feedback() : pimpl(std::make_shared<Impl>()) {}

Feedback::~Feedback() {
    shutdown();
}

void Feedback::submit(const std::string& text) {
    if (text.size() < MIN_LENGTH || text.size() > MAX_LENGTH) {
        const U64 gen = pimpl->currentGeneration();
        pimpl->setError(gen, "Feedback must be between " +
                              std::to_string(MIN_LENGTH) + " and " +
                              std::to_string(MAX_LENGTH) + " characters.");
        return;
    }

    pimpl->setSubmitting();
    const U64 gen = pimpl->currentGeneration();

#if defined(JST_OS_BROWSER)

    auto* ctx = new FetchContext{pimpl, gen, BuildFeedbackBody(text)};

    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    std::strcpy(attr.requestMethod, "POST");
    static const char* headers[] = {"Content-Type", "application/json", nullptr};
    attr.requestHeaders = headers;
    attr.requestData = ctx->body.c_str();
    attr.requestDataSize = ctx->body.size();
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.userData = ctx;
    attr.onsuccess = [](emscripten_fetch_t* fetch) {
        auto* ctx = static_cast<FetchContext*>(fetch->userData);
        std::string id;
        if (fetch->numBytes > 0) {
            try {
                auto json = nlohmann::json::parse(
                    std::string(fetch->data, fetch->numBytes));
                id = json.value("id", "");
            } catch (const std::exception&) {
                id = "";
            }
        }
        ctx->impl->setSuccess(ctx->generation, id);
        emscripten_fetch_close(fetch);
        delete ctx;
    };
    attr.onerror = [](emscripten_fetch_t* fetch) {
        auto* ctx = static_cast<FetchContext*>(fetch->userData);
        std::string message;
        if (fetch->status == 0) {
            message = "Network error: unable to reach the feedback server.";
        } else if (fetch->status == 429) {
            message = "Rate limit reached. Please try again later.";
        } else if (fetch->status == 403) {
            message = "Access denied. The feedback endpoint rejected the request.";
        } else if (fetch->status == 400) {
            message = "Invalid feedback. Please check your input and try again.";
        } else {
            message = "HTTP " + std::to_string(fetch->status);
            if (fetch->numBytes > 0) {
                message += ": " + std::string(fetch->data, fetch->numBytes);
            }
        }
        ctx->impl->setError(ctx->generation, message);
        emscripten_fetch_close(fetch);
        delete ctx;
    };
    emscripten_fetch(&attr, ENDPOINT);

#elif defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)

    auto impl = pimpl;
    const std::string body = BuildFeedbackBody(text);

    if (impl->worker.joinable()) {
        impl->worker.detach();
    }

    impl->worker = std::thread([impl, body, gen]() {
        std::string errorMessage;
        std::string resultId;

        try {
            httplib::Client cli("https://cyberether.org");
            cli.set_connection_timeout(10);
            cli.set_read_timeout(15);

            auto res = cli.Post("/api/v1/feedback", body, "application/json");

            if (!res) {
                errorMessage = "Network error: unable to reach the feedback server.";
            } else if (res->status == 202) {
                try {
                    auto responseJson = nlohmann::json::parse(res->body);
                    if (responseJson.contains("accepted") &&
                        responseJson["accepted"] == true) {
                        resultId = responseJson.value("id", "");
                    } else {
                        errorMessage = "The server did not accept the feedback.";
                    }
                } catch (const std::exception&) {
                    resultId = "";
                }
            } else if (res->status == 429) {
                errorMessage = "Rate limit reached. Please try again later.";
            } else if (res->status == 403) {
                errorMessage = "Access denied. The feedback endpoint rejected the request.";
            } else if (res->status == 400) {
                errorMessage = "Invalid feedback. Please check your input and try again.";
            } else {
                errorMessage = "HTTP " + std::to_string(res->status);
                if (!res->body.empty()) {
                    errorMessage += ": " + res->body;
                }
            }
        } catch (const std::exception& e) {
            errorMessage = std::string("Error: ") + e.what();
        } catch (...) {
            errorMessage = "Unknown error occurred while submitting feedback.";
        }

        if (errorMessage.empty()) {
            impl->setSuccess(gen, resultId);
        } else {
            impl->setError(gen, errorMessage);
        }
    });

#else

    pimpl->setError(gen, "Feedback submission is not available in this build.");

#endif
}

Feedback::Snapshot Feedback::snapshot() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->state;
}

void Feedback::clear() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    ++pimpl->generation;
    pimpl->state = {};
}

void Feedback::shutdown() {
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        pimpl->active = false;
        ++pimpl->generation;
    }

#if !defined(JST_OS_BROWSER) && defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
    if (pimpl->worker.joinable()) {
        pimpl->worker.detach();
    }
#endif
}

}  // namespace Jetstream
