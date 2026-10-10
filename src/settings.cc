#include "jetstream/settings.hh"

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

#include "jetstream/platform.hh"

namespace Jetstream {

struct Settings::Impl {
    static constexpr const char* Filename = "settings.yaml";

    static Impl& Instance();
    static Result ResolvePath(std::filesystem::path& path);
    static Result LoadFile(const std::filesystem::path& path, Settings& settings);
    static Result SaveFile(const std::filesystem::path& path, const Settings& settings);

    std::mutex mutex;
    bool loaded = false;
    std::filesystem::path path;
    Settings settings;
    U64 revision = 0;
    U64 publishedRevision = 0;
    struct Write {
        std::filesystem::path path;
        Settings settings;
        std::shared_ptr<std::promise<Result>> completion;
    };
    std::deque<Write> pending;
    std::deque<std::string> errors;
    std::condition_variable changed;
    bool stopping = false;
    bool writing = false;
    Result lastResult = Result::SUCCESS;
    std::thread worker;

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        if (worker.joinable()) worker.join();
    }

    void publish(Settings value, std::filesystem::path location) {
        settings = std::move(value);
        path = std::move(location);
        loaded = true;
    }

    void enqueue(Write write) {
        if (!worker.joinable()) worker = std::thread([this] { run(); });
        if (!write.completion && !pending.empty() && !pending.back().completion &&
            pending.back().path == write.path) {
            pending.back() = std::move(write);
        } else {
            pending.push_back(std::move(write));
        }
        changed.notify_all();
    }

    void run() {
        while (true) {
            Write write;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return stopping || !pending.empty(); });
                if (pending.empty()) return;
                write = std::move(pending.front());
                pending.pop_front();
                writing = true;
            }
            Result result = Result::ERROR;
            try {
                result = SaveFile(write.path, write.settings);
            } catch (...) {
                result = Result::ERROR;
            }
            {
                std::lock_guard lock(mutex);
                writing = false;
                lastResult = result;
                if (write.completion) {
                    write.completion->set_value(result);
                } else if (result != Result::SUCCESS) {
                    errors.push_back("Could not save settings to '" + Platform::PathToUtf8(write.path) + "'.");
                }
            }
            changed.notify_all();
        }
    }
};

Settings::Impl& Settings::Impl::Instance() {
    static Impl impl;
    return impl;
}

Result Settings::Impl::ResolvePath(std::filesystem::path& path) {
    std::string configPath;
    JST_CHECK(Platform::ConfigPath(configPath));

    path = Platform::PathFromUtf8(configPath) / Filename;
    return Result::SUCCESS;
}

Result Settings::Impl::LoadFile(const std::filesystem::path& path, Settings& settings) {
    settings = {};

    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    if (ec) {
        JST_ERROR("[SETTINGS] Failed to query settings file '{}'.", Platform::PathToUtf8(path));
        return Result::ERROR;
    }

    if (!exists) {
        return Result::SUCCESS;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        JST_ERROR("[SETTINGS] Can't open settings file '{}'.", Platform::PathToUtf8(path));
        return Result::ERROR;
    }

    const std::string yaml((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Parser::Map data;
    JST_CHECK(Parser::YamlDecode(yaml, data));
    return settings.deserialize(data);
}

Result Settings::Impl::SaveFile(const std::filesystem::path& path, const Settings& settings) {
    Parser::Map data;
    JST_CHECK(settings.serialize(data));

    std::string yaml;
    JST_CHECK(Parser::YamlEncode(data, yaml));

    return Platform::WriteFileAtomic(Platform::PathToUtf8(path), yaml);
}

Result Settings::Get(Settings& settings) {
    std::filesystem::path path;
    JST_CHECK(Impl::ResolvePath(path));

    Impl& impl = Impl::Instance();
    std::lock_guard lock(impl.mutex);
    if (!impl.loaded || impl.path != path) {
        Settings candidate;
        JST_CHECK(Impl::LoadFile(path, candidate));

        impl.publish(std::move(candidate), std::move(path));
    }

    settings = impl.settings;
    return Result::SUCCESS;
}

Result Settings::Set(const Settings& settings, bool persist) {
    std::filesystem::path path;
    JST_CHECK(Impl::ResolvePath(path));
    Settings candidate = settings;

    Impl& impl = Impl::Instance();
    std::unique_lock lock(impl.mutex);
    const auto revision = ++impl.revision;

    if (persist) {
        auto completion = std::make_shared<std::promise<Result>>();
        auto result = completion->get_future();
        impl.enqueue({path, candidate, completion});
        lock.unlock();
        JST_CHECK(result.get());
        lock.lock();
    }

    if (impl.publishedRevision < revision) {
        impl.publish(std::move(candidate), std::move(path));
        impl.publishedRevision = revision;
    }

    return Result::SUCCESS;
}

Result Settings::SetAsync(const Settings& settings) {
    std::filesystem::path path;
    JST_CHECK(Impl::ResolvePath(path));
    auto& impl = Impl::Instance();
    std::lock_guard lock(impl.mutex);
    impl.enqueue({path, settings, {}});
    impl.publish(settings, std::move(path));
    impl.publishedRevision = ++impl.revision;
    return Result::SUCCESS;
}

std::optional<std::string> Settings::TakePersistenceError() {
    auto& impl = Impl::Instance();
    std::lock_guard lock(impl.mutex);
    if (impl.errors.empty()) return std::nullopt;
    auto error = std::move(impl.errors.front());
    impl.errors.pop_front();
    return error;
}

Result Settings::Flush() {
    auto& impl = Impl::Instance();
    std::unique_lock lock(impl.mutex);
    impl.changed.wait(lock, [&] { return impl.pending.empty() && !impl.writing; });
    return impl.lastResult;
}

}  // namespace Jetstream
