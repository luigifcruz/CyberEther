#ifndef JETSTREAM_PLATFORM_HH
#define JETSTREAM_PLATFORM_HH

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"

namespace Jetstream::Platform {

//
// Paths
//

JETSTREAM_API std::filesystem::path PathFromUtf8(const std::string& path);
JETSTREAM_API std::string PathToUtf8(const std::filesystem::path& path);

JETSTREAM_API Result EnvironmentVariable(const std::string& name, std::string& value);
JETSTREAM_API Result EnvironmentPath(const std::string& name, std::filesystem::path& path);
JETSTREAM_API Result WriteEnvironmentVariable(const std::string& name, const std::optional<std::string>& value);

JETSTREAM_API Result ConfigPath(std::string& path);
JETSTREAM_API Result CachePath(std::string& path);

//
// Secrets
//

JETSTREAM_API bool SecretStoreAvailable();
JETSTREAM_API Result ReadSecret(const std::string& service, const std::string& account, std::string& value);
JETSTREAM_API Result WriteSecret(const std::string& service, const std::string& account, const std::string& value);
JETSTREAM_API Result DeleteSecret(const std::string& service, const std::string& account);

//
// Storage
//

JETSTREAM_API Result InitializePersistentStorage();

//
// Terminal
//

JETSTREAM_API bool PrepareStandardOutputForAnsi();

//
// Interrupt
//

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC) || defined(JST_OS_WINDOWS)
JETSTREAM_API bool InstallInterruptHandler(void (*handler)() noexcept);
JETSTREAM_API void UninstallInterruptHandler();
JETSTREAM_API void WriteInterruptMessage(const char* message, std::size_t size) noexcept;
[[noreturn]] JETSTREAM_API void ForceTerminate(I32 status) noexcept;
#endif

//
// Socket
//

#if !defined(JST_OS_BROWSER)
JETSTREAM_API bool ShutdownSocketRead(std::uintptr_t socket) noexcept;
JETSTREAM_API void DisableSocketSigPipe(std::uintptr_t socket) noexcept;
JETSTREAM_API void IgnoreBrokenPipe() noexcept;
JETSTREAM_API Result LocalNetworkAddress(std::string& address);

class JETSTREAM_API UdpSocket {
 public:
    struct Config {
        std::string address;
        U16 port = 0;
        std::string interfaceName;
        std::string source;
        U64 bufferBytes = 0;
        U64 timeoutMilliseconds = 0;
    };

    struct Interface {
        std::string name;
        std::vector<std::string> addresses;
    };

    static Result ListInterfaces(std::vector<Interface>& interfaces);

    UdpSocket();
    ~UdpSocket();

    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;

    Result bind(const Config& config, std::string& error);
    Result connect(const Config& config, std::string& error);
    Result receive(void* data, U64 size, U64& received);
    Result send(const void* data, U64 size);
    void close();
    bool open() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
#endif

//
// HTTP
//

class JETSTREAM_API Http {
 public:
    using Headers = std::vector<std::pair<std::string, std::string>>;

    struct Error {
        enum class Kind {
            None,
            Connection,
            Timeout,
            Tls,
            Cancelled,
            Unsupported,
        };

        Kind kind = Kind::None;
        std::string message;

        explicit operator bool() const {
            return kind != Kind::None;
        }
    };

    class JETSTREAM_API Client {
     public:
        struct Request {
            std::string method = "GET";
            std::string host;
            std::string path = "/";
            Headers headers;
            std::string body;
            U64 timeoutSeconds = 30;
            std::function<bool(I32 status, const Headers& headers)> onResponse;
            std::function<bool(std::string_view data)> onData;
            std::function<bool()> cancelled;
        };

        struct Response {
            I32 status = 0;
            Headers headers;
            std::string body;
            Error error;

            std::string header(std::string_view name) const;
        };

        static Response Send(const Request& request);
    };

    class JETSTREAM_API Server {
     public:
        struct Request {
            std::string method;
            std::string path;
            std::string query;
            Headers headers;
            std::string body;

            std::string header(std::string_view name) const;
        };

        struct Response {
            I32 status = 200;
            Headers headers;
            std::string contentType;
            std::string body;
        };

        using Handler = std::function<Response(const Request&)>;

        struct Config {
            std::string address = "127.0.0.1";
            U16 port = 0;
            U64 maxBodyBytes = 8 * 1024 * 1024;
            Handler handler;
        };

        Server();
        ~Server();

        Server(const Server&) = delete;
        Server& operator=(const Server&) = delete;

        Result start(const Config& config);
        void stop();

        bool running() const;
        U16 port() const;

     private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
};

//
// WebSocket
//

class JETSTREAM_API WebSocket {
 public:
    enum class Message {
        Text,
        Binary,
        Timeout,
        Closed,
    };

    class JETSTREAM_API Client {
     public:
        struct Config {
            std::string url;
            Http::Headers headers;
            std::vector<std::string> protocols;
            U64 connectTimeoutSeconds = 300;
            U64 readTimeoutSeconds = 0;
            U64 writeTimeoutSeconds = 5;
            U64 pingIntervalSeconds = 30;
            U64 maxMissedPongs = 0;
            U64 bufferBytes = 0;
            bool verifyCertificate = true;
        };

        struct Connection {
            I32 status = -1;
            std::string error;

            explicit operator bool() const {
                return error.empty();
            }
        };

        explicit Client(Config config);
        ~Client();

        Client(const Client&) = delete;
        Client& operator=(const Client&) = delete;

        static bool NormalizeUrl(const std::string& url, std::string& normalized);

        bool valid() const;
        Connection connect();
        Message read(std::string& payload);
        bool send(std::string_view data, Message kind = Message::Text);
        void close();
        void interrupt();
        bool open() const;

     private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    class JETSTREAM_API Server {
     public:
        struct Config {
            std::string address = "0.0.0.0";
            U16 port = 0;
            U64 writeTimeoutSeconds = 1;
            U64 pingIntervalSeconds = 30;
        };

        explicit Server(Config config);
        ~Server();

        Server(const Server&) = delete;
        Server& operator=(const Server&) = delete;

        Result start(std::string& error);
        void stop();
        U64 broadcast(std::string_view data, Message kind);
        U64 clients() const;

     private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
};

//
// Worker Thread
//

class JETSTREAM_API WorkerThread {
 public:
    WorkerThread();
    ~WorkerThread();

    WorkerThread(const WorkerThread&) = delete;
    WorkerThread& operator=(const WorkerThread&) = delete;

    void start(std::function<void()> fn);
    void join();
    bool joinable() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

//
// Process
//

JETSTREAM_API Result RunProcess(const std::string& executable,
                                const std::vector<std::string>& arguments,
                                std::string& output,
                                U64 timeoutMilliseconds = 0,
                                bool combineOutput = false,
                                std::function<void(std::string_view)> onOutput = {});

//
// Dynamic Library
//

enum class DynamicLibraryVisibility {
    Local,
    Global,
    Isolated,
};

JETSTREAM_API void* OpenDynamicLibrary(const std::string& path,
                                       DynamicLibraryVisibility visibility,
                                       std::string& error);
JETSTREAM_API void CloseDynamicLibrary(void* handle);
JETSTREAM_API void* LoadDynamicLibrarySymbol(void* handle,
                                             const char* symbol,
                                             std::string& error);

//
// Files
//

JETSTREAM_API Result WriteFileAtomic(const std::string& path, std::string_view bytes, bool ownerOnly = false);

//
// File Lock
//

class JETSTREAM_API FileLock {
 public:
    FileLock();
    ~FileLock();

    FileLock(FileLock&&) noexcept;
    FileLock& operator=(FileLock&&) noexcept;

    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

    Result acquire(const std::string& path, bool wait = true);
    void release();
    bool locked() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

//
// URL
//

JETSTREAM_API Result OpenUrl(const std::string& url);
JETSTREAM_API Result OpenFolder(const std::string& path);

//
// Dialogs
//

JETSTREAM_API Result PickFile(std::string& path,
                              const std::vector<std::string>& extensions = {},
                              std::function<void(std::string)> callback = nullptr);
JETSTREAM_API Result PickFolder(std::string& path,
                                std::function<void(std::string)> callback = nullptr);
JETSTREAM_API Result SaveFile(std::string& path,
                              const std::vector<std::string>& extensions = {},
                              std::function<void(std::string)> callback = nullptr);

JETSTREAM_API bool IsFilePending();

}  // namespace Jetstream::Platform

#endif
