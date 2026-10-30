#include "jetstream/platform.hh"
#include "jetstream/config.hh"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#if defined(JST_OS_BROWSER)
#include <atomic>
#include <deque>
#include <unordered_map>

#include <emscripten.h>
#include <emscripten/em_asm.h>

EM_JS_DEPS(jst_http_deps, "$UTF8ToString,$stringToNewUTF8,malloc,free");
#elif defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include <httplib.h>
#endif

// TODO: Port src/compositor/default/support/feedback.cc to Platform::Http.

namespace Jetstream::Platform {

namespace {

[[maybe_unused]] bool SameName(std::string_view a, std::string_view b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](unsigned char x, unsigned char y) {
        return std::tolower(x) == std::tolower(y);
    });
}

}  // namespace

std::string Http::Response::header(std::string_view name) const {
    for (const auto& [key, value] : headers) {
        if (SameName(key, name)) {
            return value;
        }
    }
    return {};
}

#if defined(JST_OS_BROWSER)

namespace {

using Clock = std::chrono::steady_clock;

constexpr auto kPollInterval = std::chrono::milliseconds(20);

struct Transfer {
    std::mutex mutex;
    std::condition_variable changed;
    std::optional<std::pair<I32, Http::Headers>> head;
    std::deque<std::string> chunks;
    std::optional<Http::Error> finished;
    Clock::time_point activity = Clock::now();
};

std::mutex transfersMutex;
std::unordered_map<U32, std::shared_ptr<Transfer>> transfers;
std::atomic<U32> nextTransfer{1};

std::shared_ptr<Transfer> FindTransfer(U32 id) {
    std::lock_guard lock(transfersMutex);
    const auto found = transfers.find(id);
    return found == transfers.end() ? nullptr : found->second;
}

Http::Headers ParseHeaders(std::string_view text) {
    Http::Headers headers;
    while (!text.empty()) {
        const auto end = text.find("\r\n");
        const auto line = text.substr(0, end);
        const auto colon = line.find(':');
        if (colon != std::string_view::npos) {
            auto value = line.substr(colon + 1);
            while (!value.empty() && value.front() == ' ') {
                value.remove_prefix(1);
            }
            headers.emplace_back(std::string(line.substr(0, colon)), std::string(value));
        }
        if (end == std::string_view::npos) {
            break;
        }
        text.remove_prefix(end + 2);
    }
    return headers;
}

std::string JoinHeaders(const Http::Headers& headers) {
    std::string text;
    for (const auto& [name, value] : headers) {
        text += name + ": " + value + "\r\n";
    }
    return text;
}

void Abort(U32 id) {
    // clang-format off
    MAIN_THREAD_EM_ASM({
        const controller = Module.jstHttp?.get($0);
        if (controller) {
            controller.abort();
        }
    }, id);
    // clang-format on
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
void jst_http_on_response(U32 id, I32 status, const char* headers) {
    if (const auto transfer = FindTransfer(id)) {
        {
            std::lock_guard lock(transfer->mutex);
            transfer->head.emplace(status, ParseHeaders(headers));
            transfer->activity = Clock::now();
        }
        transfer->changed.notify_one();
    }
}

EMSCRIPTEN_KEEPALIVE
void jst_http_on_data(U32 id, const char* data, U32 length) {
    if (const auto transfer = FindTransfer(id)) {
        {
            std::lock_guard lock(transfer->mutex);
            transfer->chunks.emplace_back(data, length);
            transfer->activity = Clock::now();
        }
        transfer->changed.notify_one();
    }
}

EMSCRIPTEN_KEEPALIVE
void jst_http_on_done(U32 id, I32 kind, const char* message) {
    if (const auto transfer = FindTransfer(id)) {
        {
            std::lock_guard lock(transfer->mutex);
            transfer->finished = Http::Error{static_cast<Http::Error::Kind>(kind), message};
        }
        transfer->changed.notify_one();
    }
}

}

Http::Response Http::Send(const Request& request) {
    Response response;
    if (request.cancelled && request.cancelled()) {
        response.error = {Error::Kind::Cancelled, "Request cancelled"};
        return response;
    }

    const U32 id = nextTransfer.fetch_add(1);
    const auto transfer = std::make_shared<Transfer>();
    {
        std::lock_guard lock(transfersMutex);
        transfers.emplace(id, transfer);
    }

    const std::string url = request.host + request.path;
    const std::string headers = JoinHeaders(request.headers);

    // clang-format off
    MAIN_THREAD_EM_ASM({
        const id = $0;
        const controller = new AbortController();
        if (!Module.jstHttp) {
            Module.jstHttp = new Map();
        }
        Module.jstHttp.set(id, controller);

        const headers = new Headers();
        for (const line of UTF8ToString($3).split('\r\n')) {
            const colon = line.indexOf(':');
            if (colon > 0) {
                try {
                    headers.append(line.slice(0, colon), line.slice(colon + 1).trim());
                } catch (error) {}
            }
        }

        const init = {};
        init.method = UTF8ToString($1);
        init.headers = headers;
        init.signal = controller.signal;
        if ($5 > 0) {
            init.body = HEAPU8.slice($4, $4 + $5);
        }

        const finish = (kind, message) => {
            Module.jstHttp.delete(id);
            const text = stringToNewUTF8(message);
            _jst_http_on_done(id, kind, text);
            _free(text);
        };

        fetch(UTF8ToString($2), init).then(async (response) => {
            let text = "";
            response.headers.forEach((value, name) => {
                text += name + ': ' + value + '\r\n';
            });
            const head = stringToNewUTF8(text);
            _jst_http_on_response(id, response.status, head);
            _free(head);

            if (response.body) {
                const reader = response.body.getReader();
                while (true) {
                    const chunk = await reader.read();
                    if (chunk.done) {
                        break;
                    }
                    const data = _malloc(chunk.value.length || 1);
                    HEAPU8.set(chunk.value, data);
                    _jst_http_on_data(id, data, chunk.value.length);
                    _free(data);
                }
            }
            finish(0, "");
        }).catch((error) => {
            const message = error && error.message ? error.message : String(error);
            finish(controller.signal.aborted ? 4 : 1, message);
        });
    }, id, request.method.c_str(), url.c_str(), headers.c_str(), request.body.data(), request.body.size());
    // clang-format on

    const auto timeout = std::chrono::seconds(request.timeoutSeconds);
    const auto cancelled = [&] {
        return request.cancelled && request.cancelled();
    };
    const auto fail = [&](Error::Kind kind, std::string message) {
        Abort(id);
        response.error = {kind, std::move(message)};
    };

    std::unique_lock lock(transfer->mutex);
    while (true) {
        transfer->changed.wait_for(lock, kPollInterval, [&] {
            return transfer->head || !transfer->chunks.empty() || transfer->finished;
        });

        if (cancelled()) {
            fail(Error::Kind::Cancelled, "Request cancelled");
            break;
        }

        if (transfer->head) {
            auto [status, headers] = std::move(*transfer->head);
            transfer->head.reset();
            response.status = status;
            response.headers = std::move(headers);

            lock.unlock();
            const bool proceed = !request.onResponse || request.onResponse(response.status, response.headers);
            lock.lock();
            if (!proceed) {
                fail(Error::Kind::Cancelled, "Request cancelled");
                break;
            }
        }

        bool proceed = true;
        while (proceed && !transfer->chunks.empty()) {
            if (cancelled()) {
                proceed = false;
                break;
            }

            auto chunk = std::move(transfer->chunks.front());
            transfer->chunks.pop_front();

            if (!request.onData) {
                response.body += chunk;
                continue;
            }

            lock.unlock();
            proceed = request.onData(chunk);
            lock.lock();
        }
        if (!proceed) {
            fail(Error::Kind::Cancelled, "Request cancelled");
            break;
        }

        if (transfer->finished && transfer->chunks.empty() && !transfer->head) {
            if (cancelled()) {
                fail(Error::Kind::Cancelled, "Request cancelled");
            } else {
                response.error = *transfer->finished;
            }
            break;
        }

        if (Clock::now() - transfer->activity >= timeout) {
            fail(Error::Kind::Timeout, "Request timed out");
            break;
        }
    }
    lock.unlock();

    std::lock_guard registry(transfersMutex);
    transfers.erase(id);
    return response;
}

#elif defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)

namespace {

using Clock = std::chrono::steady_clock;

class Cancellation {
 public:
    Cancellation(httplib::Client& client, const std::function<bool()>& cancelled) {
        if (!cancelled) {
            return;
        }

        worker = std::thread([this, &client, cancelled] {
            std::unique_lock lock(mutex);
            while (!finished) {
                if (cancelled()) {
                    client.stop();
                }
                changed.wait_for(lock, std::chrono::milliseconds(20), [this] {
                    return finished;
                });
            }
        });
    }

    ~Cancellation() {
        {
            std::lock_guard lock(mutex);
            finished = true;
        }
        changed.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
    }

 private:
    std::mutex mutex;
    std::condition_variable changed;
    bool finished = false;
    std::thread worker;
};

Http::Headers ConvertHeaders(const httplib::Headers& headers) {
    return {headers.begin(), headers.end()};
}

Http::Error ConvertError(httplib::Error error, bool idle) {
    using Kind = Http::Error::Kind;

    if (idle && (error == httplib::Error::Read || error == httplib::Error::Write)) {
        error = httplib::Error::Timeout;
    }

    Kind kind = Kind::Connection;
    switch (error) {
        case httplib::Error::Canceled:
            kind = Kind::Cancelled;
            break;
        case httplib::Error::ConnectionTimeout:
        case httplib::Error::Timeout:
            kind = Kind::Timeout;
            break;
        case httplib::Error::SSLLoadingCerts:
        case httplib::Error::SSLServerVerification:
        case httplib::Error::SSLServerHostnameVerification:
            kind = Kind::Tls;
            break;
        default:
            break;
    }
    return {kind, httplib::to_string(error)};
}

}  // namespace

Http::Response Http::Send(const Request& request) {
    Response response;
    if (request.cancelled && request.cancelled()) {
        response.error = {Error::Kind::Cancelled, httplib::to_string(httplib::Error::Canceled)};
        return response;
    }

    std::unique_ptr<httplib::Client> connection;
    try {
        connection = std::make_unique<httplib::Client>(request.host);
    } catch (const std::exception&) {
        connection.reset();
    }
    if (!connection || !connection->is_valid()) {
        response.error = {Error::Kind::Connection, "Invalid URL"};
        return response;
    }

    auto& client = *connection;
    client.set_connection_timeout(static_cast<time_t>(std::min<U64>(request.timeoutSeconds, 15)), 0);
    client.set_read_timeout(static_cast<time_t>(request.timeoutSeconds), 0);
    client.set_write_timeout(static_cast<time_t>(request.timeoutSeconds), 0);
    client.set_follow_location(true);
    client.set_socket_options([](const socket_t socket) {
        DisableSocketSigPipe(static_cast<std::uintptr_t>(socket));
    });

    const auto cancelled = [&] {
        return request.cancelled && request.cancelled();
    };

    auto activity = Clock::now();
    bool responded = false;

    httplib::Request message;
    message.method = request.method;
    message.path = request.path;
    message.headers = {request.headers.begin(), request.headers.end()};
    message.body = request.body;
    message.response_handler = [&](const httplib::Response& head) {
        activity = Clock::now();
        responded = true;
        response.status = head.status;
        response.headers = ConvertHeaders(head.headers);
        if (cancelled()) {
            return false;
        }
        return !request.onResponse || request.onResponse(response.status, response.headers);
    };
    message.content_receiver = [&](const char* data, size_t length, uint64_t, uint64_t) {
        activity = Clock::now();
        if (cancelled()) {
            return false;
        }
        if (!request.onData) {
            response.body.append(data, length);
            return true;
        }
        return request.onData(std::string_view(data, length));
    };

    Cancellation cancellation(client, request.cancelled);
    const auto result = client.send(message);
    if (cancelled()) {
        response.error = {Error::Kind::Cancelled, httplib::to_string(httplib::Error::Canceled)};
        return response;
    }
    if (!result) {
        const bool idle = Clock::now() - activity >= std::chrono::seconds(request.timeoutSeconds);
        response.error = ConvertError(result.error(), idle);
        return response;
    }

    response.status = result->status;
    response.headers = ConvertHeaders(result->headers);
    if (!responded && request.onResponse && !request.onResponse(response.status, response.headers)) {
        response.error = {Error::Kind::Cancelled, httplib::to_string(httplib::Error::Canceled)};
    }
    return response;
}

#else

Http::Response Http::Send(const Request&) {
    Response response;
    response.error = {Error::Kind::Unsupported, "This build has no HTTP support"};
    return response;
}

#endif

}  // namespace Jetstream::Platform
