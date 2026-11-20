#include "jetstream/platform.hh"

#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if defined(JST_OS_BROWSER)
#include <emscripten.h>
#include <emscripten/em_asm.h>

EM_JS_DEPS(jst_secret_deps, "$UTF8ToString,$stringToNewUTF8");
#elif defined(JST_OS_MAC) || defined(JST_OS_IOS)
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#elif defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincred.h>
#undef ERROR
#undef FATAL
#elif defined(JST_OS_LINUX)
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#endif

namespace Jetstream::Platform {

namespace {

[[maybe_unused]] bool ValidSecretText(const std::string& text) {
    return text.find('\0') == std::string::npos;
}

[[maybe_unused]] bool ValidSecretKey(const std::string& service, const std::string& account) {
    return !service.empty() && !account.empty() && ValidSecretText(service) && ValidSecretText(account);
}

[[maybe_unused]] bool ValidSecretValue(const std::string& value) {
    return !value.empty() && ValidSecretText(value);
}

}  // namespace

#if defined(JST_OS_BROWSER)

bool SecretStoreAvailable() {
    return MAIN_THREAD_EM_ASM_INT({
        const secrets = Module.cyberether?.secrets;
        return (typeof secrets?.read === 'function' &&
                typeof secrets?.write === 'function' &&
                typeof secrets?.remove === 'function') ? 1 : 0;
    }) != 0;
}

Result ReadSecret(const std::string& service, const std::string& account, std::string& value) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    char* result = static_cast<char*>(MAIN_THREAD_EM_ASM_PTR({
        const secrets = Module.cyberether?.secrets;
        if (typeof secrets?.read !== 'function' ||
            typeof secrets?.write !== 'function' ||
            typeof secrets?.remove !== 'function') {
            return 0;
        }
        try {
            const value = secrets.read(UTF8ToString($0), UTF8ToString($1));
            if (typeof value !== 'string' || value.includes('\0')) {
                return 0;
            }
            return stringToNewUTF8(value);
        } catch {
            console.error('CyberEther secret read failed.');
            return 0;
        }
    }, service.c_str(), account.c_str()));

    if (!result) {
        return Result::ERROR;
    }

    value = result;
    std::free(result);
    return Result::SUCCESS;
}

Result WriteSecret(const std::string& service, const std::string& account, const std::string& value) {
    if (!ValidSecretKey(service, account) || !ValidSecretValue(value)) {
        return Result::ERROR;
    }

    const int stored = MAIN_THREAD_EM_ASM_INT({
        const secrets = Module.cyberether?.secrets;
        if (typeof secrets?.read !== 'function' ||
            typeof secrets?.write !== 'function' ||
            typeof secrets?.remove !== 'function') {
            return 0;
        }
        try {
            return secrets.write(UTF8ToString($0), UTF8ToString($1), UTF8ToString($2)) === true ? 1 : 0;
        } catch {
            console.error('CyberEther secret write failed.');
            return 0;
        }
    }, service.c_str(), account.c_str(), value.c_str());

    return stored != 0 ? Result::SUCCESS : Result::ERROR;
}

Result DeleteSecret(const std::string& service, const std::string& account) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    const int removed = MAIN_THREAD_EM_ASM_INT({
        const secrets = Module.cyberether?.secrets;
        if (typeof secrets?.read !== 'function' ||
            typeof secrets?.write !== 'function' ||
            typeof secrets?.remove !== 'function') {
            return 0;
        }
        try {
            return secrets.remove(UTF8ToString($0), UTF8ToString($1)) === true ? 1 : 0;
        } catch {
            console.error('CyberEther secret remove failed.');
            return 0;
        }
    }, service.c_str(), account.c_str());

    return removed != 0 ? Result::SUCCESS : Result::ERROR;
}

#elif defined(JST_OS_MAC) || defined(JST_OS_IOS)

namespace {

template<typename T>
class ScopedCFRef {
 public:
    explicit ScopedCFRef(T ref = nullptr) : ref(ref) {}
    ~ScopedCFRef() {
        if (ref) {
            CFRelease(ref);
        }
    }

    ScopedCFRef(ScopedCFRef&& other) noexcept : ref(std::exchange(other.ref, nullptr)) {}
    ScopedCFRef& operator=(ScopedCFRef&& other) noexcept {
        if (this != &other) {
            if (ref) {
                CFRelease(ref);
            }
            ref = std::exchange(other.ref, nullptr);
        }
        return *this;
    }

    ScopedCFRef(const ScopedCFRef&) = delete;
    ScopedCFRef& operator=(const ScopedCFRef&) = delete;

    T get() const {
        return ref;
    }

    explicit operator bool() const {
        return ref != nullptr;
    }

 private:
    T ref;
};

ScopedCFRef<CFStringRef> MakeCFString(const std::string& value) {
    return ScopedCFRef<CFStringRef>(CFStringCreateWithBytes(kCFAllocatorDefault,
                                                            reinterpret_cast<const UInt8*>(value.data()),
                                                            static_cast<CFIndex>(value.size()),
                                                            kCFStringEncodingUTF8,
                                                            false));
}

ScopedCFRef<CFMutableDictionaryRef> MakeSecretQuery(const std::string& service, const std::string& account) {
    const auto cfService = MakeCFString(service);
    const auto cfAccount = MakeCFString(account);
    if (!cfService || !cfAccount) {
        return ScopedCFRef<CFMutableDictionaryRef>();
    }

    ScopedCFRef<CFMutableDictionaryRef> query(CFDictionaryCreateMutable(kCFAllocatorDefault,
                                                                        0,
                                                                        &kCFTypeDictionaryKeyCallBacks,
                                                                        &kCFTypeDictionaryValueCallBacks));
    if (!query) {
        return query;
    }

    CFDictionarySetValue(query.get(), kSecClass, kSecClassGenericPassword);
    CFDictionarySetValue(query.get(), kSecAttrService, cfService.get());
    CFDictionarySetValue(query.get(), kSecAttrAccount, cfAccount.get());
    return query;
}

void ReportSecretError(const char* action, OSStatus status) {
    JST_ERROR("Failed to {} secret: Security status {}.", action, static_cast<I32>(status));
}

}  // namespace

bool SecretStoreAvailable() {
    return true;
}

Result ReadSecret(const std::string& service, const std::string& account, std::string& value) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    const auto query = MakeSecretQuery(service, account);
    if (!query) {
        return Result::ERROR;
    }
    CFDictionarySetValue(query.get(), kSecReturnData, kCFBooleanTrue);
    CFDictionarySetValue(query.get(), kSecMatchLimit, kSecMatchLimitOne);

    CFTypeRef result = nullptr;
    const OSStatus status = SecItemCopyMatching(query.get(), &result);
    const ScopedCFRef<CFDataRef> data(static_cast<CFDataRef>(result));
    if (status != errSecSuccess) {
        if (status != errSecItemNotFound) {
            ReportSecretError("read", status);
        }
        return Result::ERROR;
    }
    if (!data) {
        return Result::ERROR;
    }

    value.assign(reinterpret_cast<const char*>(CFDataGetBytePtr(data.get())),
                 static_cast<std::size_t>(CFDataGetLength(data.get())));
    return Result::SUCCESS;
}

Result WriteSecret(const std::string& service, const std::string& account, const std::string& value) {
    if (!ValidSecretKey(service, account) || !ValidSecretValue(value)) {
        return Result::ERROR;
    }

    const auto query = MakeSecretQuery(service, account);
    const ScopedCFRef<CFDataRef> data(CFDataCreate(kCFAllocatorDefault,
                                                   reinterpret_cast<const UInt8*>(value.data()),
                                                   static_cast<CFIndex>(value.size())));
    if (!query || !data) {
        return Result::ERROR;
    }

    ScopedCFRef<CFMutableDictionaryRef> attributes(CFDictionaryCreateMutable(kCFAllocatorDefault,
                                                                             0,
                                                                             &kCFTypeDictionaryKeyCallBacks,
                                                                             &kCFTypeDictionaryValueCallBacks));
    if (!attributes) {
        return Result::ERROR;
    }
    CFDictionarySetValue(attributes.get(), kSecValueData, data.get());

    OSStatus status = SecItemUpdate(query.get(), attributes.get());
    if (status == errSecItemNotFound) {
        CFDictionarySetValue(query.get(), kSecValueData, data.get());
        CFDictionarySetValue(query.get(), kSecAttrAccessible, kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly);
        status = SecItemAdd(query.get(), nullptr);
    }

    if (status != errSecSuccess) {
        ReportSecretError("write", status);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DeleteSecret(const std::string& service, const std::string& account) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    const auto query = MakeSecretQuery(service, account);
    if (!query) {
        return Result::ERROR;
    }

    const OSStatus status = SecItemDelete(query.get());
    if (status != errSecSuccess && status != errSecItemNotFound) {
        ReportSecretError("delete", status);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

namespace {

constexpr const char* SecretTargetPrefix = "CyberEther/";

std::string HexEncode(const std::string& text) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(text.size() * 2);
    for (const unsigned char byte : text) {
        encoded.push_back(digits[byte >> 4]);
        encoded.push_back(digits[byte & 0x0F]);
    }
    return encoded;
}

bool MakeSecretTarget(const std::string& service, const std::string& account, std::wstring& target) {
    try {
        target = PathFromUtf8(SecretTargetPrefix + HexEncode(service) + "/" + HexEncode(account)).native();
    } catch (...) {
        return false;
    }
    return target.size() <= CRED_MAX_GENERIC_TARGET_NAME_LENGTH;
}

void ReportSecretError(const char* action) {
    JST_ERROR("Failed to {} secret: Windows error {}.", action, static_cast<U32>(GetLastError()));
}

}  // namespace

bool SecretStoreAvailable() {
    return true;
}

Result ReadSecret(const std::string& service, const std::string& account, std::string& value) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    std::wstring target;
    if (!MakeSecretTarget(service, account, target)) {
        return Result::ERROR;
    }

    PCREDENTIALW credential = nullptr;
    if (CredReadW(target.c_str(), CRED_TYPE_GENERIC, 0, &credential) == FALSE) {
        if (GetLastError() != ERROR_NOT_FOUND) {
            ReportSecretError("read");
        }
        return Result::ERROR;
    }

    value.assign(reinterpret_cast<const char*>(credential->CredentialBlob),
                 static_cast<std::size_t>(credential->CredentialBlobSize));
    CredFree(credential);
    return Result::SUCCESS;
}

Result WriteSecret(const std::string& service, const std::string& account, const std::string& value) {
    if (!ValidSecretKey(service, account) || !ValidSecretValue(value)) {
        return Result::ERROR;
    }

    if (value.size() > CRED_MAX_CREDENTIAL_BLOB_SIZE) {
        JST_ERROR("Failed to write secret: value exceeds the Credential Manager size limit.");
        return Result::ERROR;
    }

    std::wstring target;
    std::wstring user;
    if (!MakeSecretTarget(service, account, target)) {
        return Result::ERROR;
    }
    try {
        user = PathFromUtf8(account).native();
    } catch (...) {
        return Result::ERROR;
    }

    std::vector<BYTE> blob(value.begin(), value.end());

    CREDENTIALW credential = {};
    credential.Type = CRED_TYPE_GENERIC;
    credential.TargetName = target.data();
    credential.UserName = user.data();
    credential.CredentialBlobSize = static_cast<DWORD>(blob.size());
    credential.CredentialBlob = blob.empty() ? nullptr : blob.data();
    credential.Persist = CRED_PERSIST_LOCAL_MACHINE;

    if (CredWriteW(&credential, 0) == FALSE) {
        ReportSecretError("write");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DeleteSecret(const std::string& service, const std::string& account) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    std::wstring target;
    if (!MakeSecretTarget(service, account, target)) {
        return Result::ERROR;
    }

    if (CredDeleteW(target.c_str(), CRED_TYPE_GENERIC, 0) == FALSE && GetLastError() != ERROR_NOT_FOUND) {
        ReportSecretError("delete");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

#elif defined(JST_OS_LINUX)

namespace {

struct DBusConnection;
struct DBusMessage;

struct DBusError {
    const char* name;
    const char* message;
    unsigned int dummy1 : 1;
    unsigned int dummy2 : 1;
    unsigned int dummy3 : 1;
    unsigned int dummy4 : 1;
    unsigned int dummy5 : 1;
    void* padding1;
};

struct DBusMessageIter {
    void* dummy1;
    void* dummy2;
    std::uint32_t dummy3;
    int dummy4;
    int dummy5;
    int dummy6;
    int dummy7;
    int dummy8;
    int dummy9;
    int dummy10;
    int dummy11;
    int pad1;
    void* pad2;
    void* pad3;
};

using DBusBool = std::uint32_t;

constexpr int DBusBusSession = 0;
constexpr int DBusTypeInvalid = 0;
constexpr int DBusTypeArray = 'a';
constexpr int DBusTypeBoolean = 'b';
constexpr int DBusTypeByte = 'y';
constexpr int DBusTypeDictEntry = 'e';
constexpr int DBusTypeObjectPath = 'o';
constexpr int DBusTypeString = 's';
constexpr int DBusTypeStruct = 'r';
constexpr int DBusTypeUint32 = 'u';
constexpr int DBusTypeVariant = 'v';

constexpr std::size_t DBusMaximumArrayBytes = 64 * 1024 * 1024;
constexpr std::size_t SecretMaximumKeyBytes = 16 * 1024 * 1024;

constexpr int CallTimeoutMilliseconds = 25000;
constexpr auto PromptTimeout = std::chrono::seconds(120);
constexpr int PromptPollMilliseconds = 250;

constexpr const char* SecretBusName = "org.freedesktop.secrets";
constexpr const char* SecretServicePath = "/org/freedesktop/secrets";
constexpr const char* SecretDefaultCollectionPath = "/org/freedesktop/secrets/aliases/default";
constexpr const char* SecretServiceInterface = "org.freedesktop.Secret.Service";
constexpr const char* SecretCollectionInterface = "org.freedesktop.Secret.Collection";
constexpr const char* SecretItemInterface = "org.freedesktop.Secret.Item";
constexpr const char* SecretPromptInterface = "org.freedesktop.Secret.Prompt";
constexpr const char* PropertiesInterface = "org.freedesktop.DBus.Properties";
constexpr const char* SecretNoObject = "/";
constexpr const char* SecretSessionAlgorithm = "plain";
constexpr const char* SecretContentType = "text/plain";
constexpr const char* SecretLegacyContentType = "application/octet-stream";
constexpr const char* SecretCollectionDefault = "default";
constexpr const char* SecretCollectionLabel = "Default keyring";

constexpr const char* SecretSchemaAttribute = "xdg:schema";
constexpr const char* SecretSchemaName = "ltd.luigi.CyberEther.Secret";
constexpr const char* SecretServiceAttribute = "service";
constexpr const char* SecretAccountAttribute = "account";

constexpr const char* MissingObjectErrors[] = {
    "org.freedesktop.Secret.Error.NoSuchObject",
    "org.freedesktop.DBus.Error.UnknownObject",
    "org.freedesktop.DBus.Error.UnknownMethod",
};

enum class PromptResult {
    Any,
    Object,
    Objects,
};

struct Libdbus {
    void* handle = nullptr;
    DBusBool (*threadsInitDefault)() = nullptr;
    void (*errorInit)(DBusError*) = nullptr;
    void (*errorFree)(DBusError*) = nullptr;
    DBusBool (*errorIsSet)(const DBusError*) = nullptr;
    DBusBool (*validateUtf8)(const char*, DBusError*) = nullptr;
    DBusConnection* (*busGetPrivate)(int, DBusError*) = nullptr;
    DBusBool (*busNameHasOwner)(DBusConnection*, const char*, DBusError*) = nullptr;
    void (*busAddMatch)(DBusConnection*, const char*, DBusError*) = nullptr;
    void (*busRemoveMatch)(DBusConnection*, const char*, DBusError*) = nullptr;
    void (*connectionSetExitOnDisconnect)(DBusConnection*, DBusBool) = nullptr;
    void (*connectionClose)(DBusConnection*) = nullptr;
    void (*connectionUnref)(DBusConnection*) = nullptr;
    DBusBool (*connectionGetIsConnected)(DBusConnection*) = nullptr;
    DBusMessage* (*connectionSendWithReplyAndBlock)(DBusConnection*, DBusMessage*, int, DBusError*) = nullptr;
    DBusBool (*connectionReadWrite)(DBusConnection*, int) = nullptr;
    DBusMessage* (*connectionPopMessage)(DBusConnection*) = nullptr;
    DBusMessage* (*messageNewMethodCall)(const char*, const char*, const char*, const char*) = nullptr;
    void (*messageUnref)(DBusMessage*) = nullptr;
    void (*messageSetAutoStart)(DBusMessage*, DBusBool) = nullptr;
    DBusBool (*messageIsSignal)(DBusMessage*, const char*, const char*) = nullptr;
    DBusBool (*messageHasSignature)(DBusMessage*, const char*) = nullptr;
    const char* (*messageGetPath)(DBusMessage*) = nullptr;
    const char* (*messageGetSender)(DBusMessage*) = nullptr;
    DBusBool (*iterInit)(DBusMessage*, DBusMessageIter*) = nullptr;
    void (*iterInitAppend)(DBusMessage*, DBusMessageIter*) = nullptr;
    DBusBool (*iterAppendBasic)(DBusMessageIter*, int, const void*) = nullptr;
    DBusBool (*iterAppendFixedArray)(DBusMessageIter*, int, const void*, int) = nullptr;
    DBusBool (*iterOpenContainer)(DBusMessageIter*, int, const char*, DBusMessageIter*) = nullptr;
    DBusBool (*iterCloseContainer)(DBusMessageIter*, DBusMessageIter*) = nullptr;
    void (*iterAbandonContainerIfOpen)(DBusMessageIter*, DBusMessageIter*) = nullptr;
    int (*iterGetArgType)(DBusMessageIter*) = nullptr;
    int (*iterGetElementType)(DBusMessageIter*) = nullptr;
    void (*iterGetBasic)(DBusMessageIter*, void*) = nullptr;
    void (*iterGetFixedArray)(DBusMessageIter*, void*, int*) = nullptr;
    void (*iterRecurse)(DBusMessageIter*, DBusMessageIter*) = nullptr;
    DBusBool (*iterNext)(DBusMessageIter*) = nullptr;
};

template<typename F>
bool LoadSymbol(void* handle, const char* name, F& target) {
    std::string error;
    target = reinterpret_cast<F>(LoadDynamicLibrarySymbol(handle, name, error));
    return target != nullptr;
}

const Libdbus& GetLibdbus() {
    static Libdbus library;
    static std::once_flag flag;
    std::call_once(flag, [] {
        std::string error;
        library.handle = OpenDynamicLibrary("libdbus-1.so.3", DynamicLibraryVisibility::Local, error);
        if (!library.handle) {
            return;
        }

        const bool loaded = LoadSymbol(library.handle, "dbus_threads_init_default", library.threadsInitDefault) &&
                            LoadSymbol(library.handle, "dbus_error_init", library.errorInit) &&
                            LoadSymbol(library.handle, "dbus_error_free", library.errorFree) &&
                            LoadSymbol(library.handle, "dbus_error_is_set", library.errorIsSet) &&
                            LoadSymbol(library.handle, "dbus_validate_utf8", library.validateUtf8) &&
                            LoadSymbol(library.handle, "dbus_bus_get_private", library.busGetPrivate) &&
                            LoadSymbol(library.handle, "dbus_bus_name_has_owner", library.busNameHasOwner) &&
                            LoadSymbol(library.handle, "dbus_bus_add_match", library.busAddMatch) &&
                            LoadSymbol(library.handle, "dbus_bus_remove_match", library.busRemoveMatch) &&
                            LoadSymbol(library.handle, "dbus_connection_set_exit_on_disconnect",
                                       library.connectionSetExitOnDisconnect) &&
                            LoadSymbol(library.handle, "dbus_connection_close", library.connectionClose) &&
                            LoadSymbol(library.handle, "dbus_connection_unref", library.connectionUnref) &&
                            LoadSymbol(library.handle, "dbus_connection_get_is_connected",
                                       library.connectionGetIsConnected) &&
                            LoadSymbol(library.handle, "dbus_connection_send_with_reply_and_block",
                                       library.connectionSendWithReplyAndBlock) &&
                            LoadSymbol(library.handle, "dbus_connection_read_write", library.connectionReadWrite) &&
                            LoadSymbol(library.handle, "dbus_connection_pop_message", library.connectionPopMessage) &&
                            LoadSymbol(library.handle, "dbus_message_new_method_call", library.messageNewMethodCall) &&
                            LoadSymbol(library.handle, "dbus_message_unref", library.messageUnref) &&
                            LoadSymbol(library.handle, "dbus_message_set_auto_start", library.messageSetAutoStart) &&
                            LoadSymbol(library.handle, "dbus_message_is_signal", library.messageIsSignal) &&
                            LoadSymbol(library.handle, "dbus_message_has_signature", library.messageHasSignature) &&
                            LoadSymbol(library.handle, "dbus_message_get_path", library.messageGetPath) &&
                            LoadSymbol(library.handle, "dbus_message_get_sender", library.messageGetSender) &&
                            LoadSymbol(library.handle, "dbus_message_iter_init", library.iterInit) &&
                            LoadSymbol(library.handle, "dbus_message_iter_init_append", library.iterInitAppend) &&
                            LoadSymbol(library.handle, "dbus_message_iter_append_basic", library.iterAppendBasic) &&
                            LoadSymbol(library.handle, "dbus_message_iter_append_fixed_array",
                                       library.iterAppendFixedArray) &&
                            LoadSymbol(library.handle, "dbus_message_iter_open_container", library.iterOpenContainer) &&
                            LoadSymbol(library.handle, "dbus_message_iter_close_container",
                                       library.iterCloseContainer) &&
                            LoadSymbol(library.handle, "dbus_message_iter_abandon_container_if_open",
                                       library.iterAbandonContainerIfOpen) &&
                            LoadSymbol(library.handle, "dbus_message_iter_get_arg_type", library.iterGetArgType) &&
                            LoadSymbol(library.handle, "dbus_message_iter_get_element_type",
                                       library.iterGetElementType) &&
                            LoadSymbol(library.handle, "dbus_message_iter_get_basic", library.iterGetBasic) &&
                            LoadSymbol(library.handle, "dbus_message_iter_get_fixed_array", library.iterGetFixedArray) &&
                            LoadSymbol(library.handle, "dbus_message_iter_recurse", library.iterRecurse) &&
                            LoadSymbol(library.handle, "dbus_message_iter_next", library.iterNext);
        if (!loaded || !library.threadsInitDefault()) {
            CloseDynamicLibrary(library.handle);
            library = Libdbus{};
        }
    });
    return library;
}

std::mutex& SecretMutex() {
    static std::mutex mutex;
    return mutex;
}

DBusConnection*& SharedConnection() {
    static DBusConnection* connection = nullptr;
    return connection;
}

std::atomic<pid_t>& SharedConnectionPid() {
    static std::atomic<pid_t> pid{0};
    return pid;
}

bool InheritedConnection() {
    const pid_t pid = SharedConnectionPid().load();
    return pid != 0 && pid != getpid();
}

bool RejectInheritedConnection(const char* action) {
    if (!InheritedConnection()) {
        return false;
    }
    JST_ERROR("Failed to {} secret: the Secret Service connection belongs to the parent process.", action);
    return true;
}

void RetireSharedConnection(const Libdbus& library) {
    auto& shared = SharedConnection();
    if (shared) {
        library.connectionClose(shared);
        library.connectionUnref(shared);
        shared = nullptr;
    }
    SharedConnectionPid().store(0);
}

struct MessageDeleter {
    void operator()(DBusMessage* message) const {
        GetLibdbus().messageUnref(message);
    }
};

using MessagePtr = std::unique_ptr<DBusMessage, MessageDeleter>;

class ScopedError {
 public:
    explicit ScopedError(const Libdbus& library) : library(library) {
        library.errorInit(&error);
    }
    ~ScopedError() {
        library.errorFree(&error);
    }

    ScopedError(const ScopedError&) = delete;
    ScopedError& operator=(const ScopedError&) = delete;

    DBusError* get() {
        return &error;
    }

    bool set() const {
        return library.errorIsSet(&error);
    }

    const char* name() const {
        return error.name ? error.name : "unknown";
    }

    const char* message() const {
        return error.message ? error.message : "unknown error";
    }

 private:
    const Libdbus& library;
    DBusError error = {};
};

class ScopedContainer {
 public:
    ScopedContainer(const Libdbus& library, DBusMessageIter* parent, int type, const char* signature)
        : library(library), parent(parent) {
        opened = library.iterOpenContainer(parent, type, signature, &iter) != 0;
    }
    ~ScopedContainer() {
        library.iterAbandonContainerIfOpen(parent, &iter);
    }

    ScopedContainer(const ScopedContainer&) = delete;
    ScopedContainer& operator=(const ScopedContainer&) = delete;

    bool ok() const {
        return opened;
    }

    DBusMessageIter* get() {
        return &iter;
    }

    bool close() {
        return opened && library.iterCloseContainer(parent, &iter) != 0;
    }

 private:
    const Libdbus& library;
    DBusMessageIter* parent;
    DBusMessageIter iter = {};
    bool opened = false;
};

struct SecretItems {
    std::vector<std::string> unlocked;
    std::vector<std::string> locked;
};

bool ContainsPath(const std::vector<std::string>& paths, const std::string& path) {
    return std::find(paths.begin(), paths.end(), path) != paths.end();
}

class SecretService {
 public:
    SecretService(const Libdbus& library, const char* action) : library(library), action(action) {}
    ~SecretService() {
        if (connection && !session.empty() && session != SecretNoObject) {
            closeSession();
        }
    }

    SecretService(const SecretService&) = delete;
    SecretService& operator=(const SecretService&) = delete;

    bool connect(bool report = true) {
        auto& shared = SharedConnection();
        if (shared) {
            library.connectionReadWrite(shared, 0);
            if (!library.connectionGetIsConnected(shared)) {
                RetireSharedConnection(library);
            }
        }

        if (!shared) {
            ScopedError error(library);
            shared = library.busGetPrivate(DBusBusSession, error.get());
            if (!shared) {
                if (report) {
                    fail(error);
                }
                return false;
            }
            library.connectionSetExitOnDisconnect(shared, 0);
            SharedConnectionPid().store(getpid());
        }

        while (MessagePtr stale{library.connectionPopMessage(shared)}) {
        }
        connection = shared;
        return true;
    }

    bool resolveOwner() {
        if (!serviceOwned() && !activate()) {
            return false;
        }

        MessagePtr message(library.messageNewMethodCall("org.freedesktop.DBus", "/org/freedesktop/DBus",
                                                        "org.freedesktop.DBus", "GetNameOwner"));
        if (!message) {
            return outOfMemory();
        }

        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendBasic(&args, DBusTypeString, SecretBusName)) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }

        DBusMessageIter iter;
        const char* name = nullptr;
        if (!library.messageHasSignature(reply.get(), "s") || !library.iterInit(reply.get(), &iter)) {
            return unexpectedReply();
        }
        library.iterGetBasic(&iter, &name);
        if (!name || name[0] != ':') {
            return unexpectedReply();
        }
        owner = name;
        return true;
    }

    bool serviceOwned() {
        ScopedError error(library);
        const bool owned = library.busNameHasOwner(connection, SecretBusName, error.get()) != 0;
        return owned && !error.set();
    }

    bool openSession() {
        auto message = call(SecretServicePath, SecretServiceInterface, "OpenSession");
        if (!message) {
            return false;
        }

        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendBasic(&args, DBusTypeString, SecretSessionAlgorithm) || !appendStringVariant(&args, "")) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }

        DBusMessageIter iter;
        DBusMessageIter output;
        if (!library.messageHasSignature(reply.get(), "vo") || !library.iterInit(reply.get(), &iter)) {
            retire();
            return unexpectedReply();
        }
        library.iterRecurse(&iter, &output);
        std::string path;
        library.iterNext(&iter);
        if (!emptySessionOutput(&output) || !readObjectPath(&iter, path) || path.empty() ||
            path == SecretNoObject) {
            retire();
            return unexpectedReply();
        }
        session = path;
        return true;
    }

    bool search(const std::string& service, const std::string& account, SecretItems& items) {
        auto message = call(SecretServicePath, SecretServiceInterface, "SearchItems");
        if (!message) {
            return false;
        }

        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendAttributes(&args, service, account)) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }

        DBusMessageIter iter;
        if (!library.messageHasSignature(reply.get(), "aoao") || !library.iterInit(reply.get(), &iter) ||
            !readObjectPaths(&iter, items.unlocked) || !library.iterNext(&iter) ||
            !readObjectPaths(&iter, items.locked)) {
            return unexpectedReply();
        }
        return true;
    }

    bool unlock(const std::vector<std::string>& objects, std::vector<std::string>& unlocked) {
        auto message = call(SecretServicePath, SecretServiceInterface, "Unlock");
        if (!message) {
            return false;
        }

        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendObjectPaths(&args, objects)) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }

        DBusMessageIter iter;
        std::string promptPath;
        if (!library.messageHasSignature(reply.get(), "aoo") || !library.iterInit(reply.get(), &iter) ||
            !readObjectPaths(&iter, unlocked) || !library.iterNext(&iter) ||
            !readObjectPath(&iter, promptPath)) {
            return unexpectedReply();
        }

        return prompt(promptPath, PromptResult::Objects, unlocked);
    }

    bool getSecret(const std::string& item, std::string& value) {
        auto message = call(item.c_str(), SecretItemInterface, "GetSecret");
        if (!message) {
            return false;
        }

        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendBasic(&args, DBusTypeObjectPath, session.c_str())) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }

        DBusMessageIter iter;
        DBusMessageIter secret;
        DBusMessageIter parameters;
        DBusMessageIter bytes;
        if (!library.messageHasSignature(reply.get(), "(oayays)") || !library.iterInit(reply.get(), &iter)) {
            return unexpectedReply();
        }
        library.iterRecurse(&iter, &secret);
        std::string secretSession;
        readObjectPath(&secret, secretSession);
        library.iterNext(&secret);
        library.iterRecurse(&secret, &parameters);
        if (secretSession != session || library.iterGetArgType(&parameters) != DBusTypeInvalid) {
            return unexpectedReply();
        }
        library.iterNext(&secret);
        library.iterRecurse(&secret, &bytes);

        const unsigned char* data = nullptr;
        int size = 0;
        if (library.iterGetArgType(&bytes) == DBusTypeByte) {
            library.iterGetFixedArray(&bytes, &data, &size);
        }
        if (size < 0 || (size > 0 && !data)) {
            return unexpectedReply();
        }

        const char* contentType = nullptr;
        library.iterNext(&secret);
        library.iterGetBasic(&secret, &contentType);

        const std::string secretValue = size > 0 ? std::string(reinterpret_cast<const char*>(data),
                                                               static_cast<std::size_t>(size))
                                                 : std::string();
        if (!ValidSecretText(secretValue)) {
            JST_ERROR("Failed to {} secret: the stored value contains a null byte.", action);
            return false;
        }

        const std::string type = contentType ? contentType : "";
        const bool text = type == SecretContentType ||
                          (type == SecretLegacyContentType && library.validateUtf8(secretValue.c_str(), nullptr));
        if (!text) {
            JST_ERROR("Failed to {} secret: the stored value is not text.", action);
            return false;
        }
        value = secretValue;
        return true;
    }

    bool ensureDefaultCollection() {
        std::string collection;
        if (!readAlias(collection)) {
            return false;
        }
        if (collection != SecretNoObject) {
            return true;
        }

        auto message = call(SecretServicePath, SecretServiceInterface, "CreateCollection");
        if (!message) {
            return false;
        }

        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendCollectionProperties(&args) || !appendBasic(&args, DBusTypeString, SecretCollectionDefault)) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }

        DBusMessageIter iter;
        std::string promptPath;
        if (!library.messageHasSignature(reply.get(), "oo") || !library.iterInit(reply.get(), &iter) ||
            !readObjectPath(&iter, collection) || !library.iterNext(&iter) ||
            !readObjectPath(&iter, promptPath)) {
            return unexpectedReply();
        }

        if (promptPath == SecretNoObject) {
            if (collection == SecretNoObject) {
                return unexpectedReply();
            }
        } else {
            std::vector<std::string> created;
            if (!prompt(promptPath, PromptResult::Object, created)) {
                return false;
            }
        }

        if (!readAlias(collection)) {
            return false;
        }
        if (collection == SecretNoObject) {
            JST_ERROR("Failed to {} secret: the default collection could not be created.", action);
            return false;
        }
        return true;
    }

    bool unlockDefaultCollection() {
        bool locked = false;
        if (!collectionLocked(locked)) {
            return false;
        }
        if (!locked) {
            return true;
        }

        std::vector<std::string> unlocked;
        if (!unlock({SecretDefaultCollectionPath}, unlocked) || !collectionLocked(locked)) {
            return false;
        }
        if (locked) {
            JST_ERROR("Failed to {} secret: the default collection is still locked.", action);
            return false;
        }
        return true;
    }

    bool createItem(const std::string& service, const std::string& account, const std::string& value) {
        auto message = call(SecretDefaultCollectionPath, SecretCollectionInterface, "CreateItem");
        if (!message) {
            return false;
        }

        const DBusBool replace = 1;
        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendItemProperties(&args, service, account) || !appendSecret(&args, value) ||
            !library.iterAppendBasic(&args, DBusTypeBoolean, &replace)) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }

        DBusMessageIter iter;
        std::string item;
        std::string promptPath;
        if (!library.messageHasSignature(reply.get(), "oo") || !library.iterInit(reply.get(), &iter) ||
            !readObjectPath(&iter, item) || !library.iterNext(&iter) || !readObjectPath(&iter, promptPath)) {
            return unexpectedReply();
        }

        if (promptPath == SecretNoObject) {
            return item != SecretNoObject || unexpectedReply();
        }

        std::vector<std::string> created;
        return prompt(promptPath, PromptResult::Object, created);
    }

    bool deleteItem(const std::string& item) {
        auto message = call(item.c_str(), SecretItemInterface, "Delete");
        if (!message) {
            return false;
        }

        bool missing = false;
        const auto reply = send(std::move(message), &missing);
        if (!reply) {
            return missing;
        }

        DBusMessageIter iter;
        std::string promptPath;
        if (!library.messageHasSignature(reply.get(), "o") || !library.iterInit(reply.get(), &iter) ||
            !readObjectPath(&iter, promptPath)) {
            return unexpectedReply();
        }

        std::vector<std::string> results;
        return prompt(promptPath, PromptResult::Any, results);
    }

 private:
    const Libdbus& library;
    const char* action;
    DBusConnection* connection = nullptr;
    std::string owner;
    std::string session;

    bool activate() {
        MessagePtr message(library.messageNewMethodCall("org.freedesktop.DBus", "/org/freedesktop/DBus",
                                                        "org.freedesktop.DBus", "StartServiceByName"));
        if (!message) {
            return outOfMemory();
        }

        const std::uint32_t flags = 0;
        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendBasic(&args, DBusTypeString, SecretBusName) ||
            !library.iterAppendBasic(&args, DBusTypeUint32, &flags)) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }
        if (!library.messageHasSignature(reply.get(), "u")) {
            return unexpectedReply();
        }
        return true;
    }

    bool emptySessionOutput(DBusMessageIter* output) {
        if (library.iterGetArgType(output) == DBusTypeString) {
            const char* value = nullptr;
            library.iterGetBasic(output, &value);
            return value && value[0] == '\0';
        }
        if (library.iterGetArgType(output) == DBusTypeArray && library.iterGetElementType(output) == DBusTypeByte) {
            DBusMessageIter bytes;
            library.iterRecurse(output, &bytes);
            return library.iterGetArgType(&bytes) == DBusTypeInvalid;
        }
        return false;
    }

    void retire() {
        RetireSharedConnection(library);
        connection = nullptr;
    }

    bool fail(const ScopedError& error) {
        JST_ERROR("Failed to {} secret: {} ({}).", action, error.message(), error.name());
        return false;
    }

    bool outOfMemory() {
        JST_ERROR("Failed to {} secret: out of memory while building a D-Bus message.", action);
        return false;
    }

    bool unexpectedReply() {
        JST_ERROR("Failed to {} secret: unexpected reply from the Secret Service.", action);
        return false;
    }

    MessagePtr call(const char* path, const char* interface, const char* method) {
        MessagePtr message(library.messageNewMethodCall(owner.c_str(), path, interface, method));
        if (!message) {
            outOfMemory();
            return message;
        }
        library.messageSetAutoStart(message.get(), 0);
        return message;
    }

    MessagePtr send(MessagePtr message, bool* missing = nullptr) {
        if (!connection) {
            JST_ERROR("Failed to {} secret: the Secret Service connection was closed.", action);
            return nullptr;
        }
        ScopedError error(library);
        MessagePtr reply(library.connectionSendWithReplyAndBlock(connection, message.get(),
                                                                 CallTimeoutMilliseconds, error.get()));
        if (!reply) {
            if (missing && error.set()) {
                for (const char* name : MissingObjectErrors) {
                    if (std::string(error.name()) == name) {
                        *missing = true;
                        return reply;
                    }
                }
            }
            fail(error);
        }
        return reply;
    }

    bool appendBasic(DBusMessageIter* iter, int type, const char* value) {
        return library.iterAppendBasic(iter, type, &value) != 0;
    }

    bool appendStringVariant(DBusMessageIter* iter, const char* value) {
        ScopedContainer variant(library, iter, DBusTypeVariant, "s");
        return variant.ok() && appendBasic(variant.get(), DBusTypeString, value) && variant.close();
    }

    bool appendStringPair(DBusMessageIter* dict, const char* key, const char* value) {
        ScopedContainer entry(library, dict, DBusTypeDictEntry, nullptr);
        return entry.ok() && appendBasic(entry.get(), DBusTypeString, key) &&
               appendBasic(entry.get(), DBusTypeString, value) && entry.close();
    }

    bool appendAttributes(DBusMessageIter* iter, const std::string& service, const std::string& account) {
        ScopedContainer dict(library, iter, DBusTypeArray, "{ss}");
        return dict.ok() && appendStringPair(dict.get(), SecretSchemaAttribute, SecretSchemaName) &&
               appendStringPair(dict.get(), SecretServiceAttribute, service.c_str()) &&
               appendStringPair(dict.get(), SecretAccountAttribute, account.c_str()) && dict.close();
    }

    bool appendStringProperty(DBusMessageIter* dict, const char* key, const char* value) {
        ScopedContainer entry(library, dict, DBusTypeDictEntry, nullptr);
        return entry.ok() && appendBasic(entry.get(), DBusTypeString, key) &&
               appendStringVariant(entry.get(), value) && entry.close();
    }

    bool appendAttributesProperty(DBusMessageIter* dict, const std::string& service, const std::string& account) {
        ScopedContainer entry(library, dict, DBusTypeDictEntry, nullptr);
        if (!entry.ok() || !appendBasic(entry.get(), DBusTypeString, "org.freedesktop.Secret.Item.Attributes")) {
            return false;
        }
        ScopedContainer variant(library, entry.get(), DBusTypeVariant, "a{ss}");
        return variant.ok() && appendAttributes(variant.get(), service, account) && variant.close() &&
               entry.close();
    }

    bool appendCollectionProperties(DBusMessageIter* iter) {
        ScopedContainer properties(library, iter, DBusTypeArray, "{sv}");
        return properties.ok() &&
               appendStringProperty(properties.get(), "org.freedesktop.Secret.Collection.Label",
                                    SecretCollectionLabel) &&
               properties.close();
    }

    bool appendItemProperties(DBusMessageIter* iter, const std::string& service, const std::string& account) {
        const std::string label = service + " (" + account + ")";
        ScopedContainer properties(library, iter, DBusTypeArray, "{sv}");
        return properties.ok() &&
               appendStringProperty(properties.get(), "org.freedesktop.Secret.Item.Label", label.c_str()) &&
               appendAttributesProperty(properties.get(), service, account) && properties.close();
    }

    bool appendBytes(DBusMessageIter* iter, const std::string& value) {
        const auto* data = reinterpret_cast<const unsigned char*>(value.data());
        ScopedContainer bytes(library, iter, DBusTypeArray, "y");
        return bytes.ok() &&
               (value.empty() ||
                library.iterAppendFixedArray(bytes.get(), DBusTypeByte, &data, static_cast<int>(value.size()))) &&
               bytes.close();
    }

    bool appendSecret(DBusMessageIter* iter, const std::string& value) {
        ScopedContainer secret(library, iter, DBusTypeStruct, nullptr);
        return secret.ok() && appendBasic(secret.get(), DBusTypeObjectPath, session.c_str()) &&
               appendBytes(secret.get(), "") && appendBytes(secret.get(), value) &&
               appendBasic(secret.get(), DBusTypeString, SecretContentType) && secret.close();
    }

    bool appendObjectPaths(DBusMessageIter* iter, const std::vector<std::string>& paths) {
        ScopedContainer array(library, iter, DBusTypeArray, "o");
        if (!array.ok()) {
            return false;
        }
        for (const auto& path : paths) {
            if (!appendBasic(array.get(), DBusTypeObjectPath, path.c_str())) {
                return false;
            }
        }
        return array.close();
    }

    bool readObjectPath(DBusMessageIter* iter, std::string& path) {
        if (library.iterGetArgType(iter) != DBusTypeObjectPath) {
            return false;
        }
        const char* value = nullptr;
        library.iterGetBasic(iter, &value);
        path = value ? value : "";
        return true;
    }

    bool readObjectPaths(DBusMessageIter* iter, std::vector<std::string>& paths) {
        if (library.iterGetArgType(iter) != DBusTypeArray ||
            library.iterGetElementType(iter) != DBusTypeObjectPath) {
            return false;
        }
        DBusMessageIter array;
        library.iterRecurse(iter, &array);
        while (library.iterGetArgType(&array) == DBusTypeObjectPath) {
            std::string path;
            readObjectPath(&array, path);
            paths.push_back(std::move(path));
            library.iterNext(&array);
        }
        return library.iterGetArgType(&array) == DBusTypeInvalid;
    }

    bool readAlias(std::string& collection) {
        auto message = call(SecretServicePath, SecretServiceInterface, "ReadAlias");
        if (!message) {
            return false;
        }

        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendBasic(&args, DBusTypeString, SecretCollectionDefault)) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }

        DBusMessageIter iter;
        if (!library.messageHasSignature(reply.get(), "o") || !library.iterInit(reply.get(), &iter) ||
            !readObjectPath(&iter, collection)) {
            return unexpectedReply();
        }
        return true;
    }

    bool collectionLocked(bool& locked) {
        auto message = call(SecretDefaultCollectionPath, PropertiesInterface, "Get");
        if (!message) {
            return false;
        }

        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendBasic(&args, DBusTypeString, SecretCollectionInterface) ||
            !appendBasic(&args, DBusTypeString, "Locked")) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply) {
            return false;
        }

        DBusMessageIter iter;
        DBusMessageIter variant;
        if (!library.messageHasSignature(reply.get(), "v") || !library.iterInit(reply.get(), &iter)) {
            return unexpectedReply();
        }
        library.iterRecurse(&iter, &variant);
        if (library.iterGetArgType(&variant) != DBusTypeBoolean) {
            return unexpectedReply();
        }
        DBusBool value = 0;
        library.iterGetBasic(&variant, &value);
        locked = value != 0;
        return true;
    }

    bool prompt(const std::string& path, PromptResult kind, std::vector<std::string>& results) {
        if (path == SecretNoObject) {
            return true;
        }

        if (!connection) {
            JST_ERROR("Failed to {} secret: the Secret Service connection was closed.", action);
            return false;
        }

        const std::string sender = owner;

        const std::vector<std::string> rules = {
            "type='signal',sender='" + sender + "',interface='" + std::string(SecretPromptInterface) +
                "',member='Completed',path='" + path + "'",
            "type='signal',sender='org.freedesktop.DBus',interface='org.freedesktop.DBus',"
            "member='NameOwnerChanged',arg0='" + sender + "'",
        };

        std::size_t added = 0;
        bool completed = false;
        for (; added < rules.size(); ++added) {
            ScopedError error(library);
            library.busAddMatch(connection, rules[added].c_str(), error.get());
            if (error.set()) {
                fail(error);
                break;
            }
        }
        if (added == rules.size()) {
            completed = runPrompt(path, sender, kind, results);
        }

        for (std::size_t index = 0; connection && index < added; ++index) {
            ScopedError error(library);
            library.busRemoveMatch(connection, rules[index].c_str(), error.get());
        }
        return completed;
    }

    bool runPrompt(const std::string& path,
                   const std::string& sender,
                   PromptResult kind,
                   std::vector<std::string>& results) {
        auto message = call(path.c_str(), SecretPromptInterface, "Prompt");
        if (!message) {
            return false;
        }

        DBusMessageIter args;
        library.iterInitAppend(message.get(), &args);
        if (!appendBasic(&args, DBusTypeString, "")) {
            return outOfMemory();
        }

        const auto reply = send(std::move(message));
        if (!reply || !library.messageHasSignature(reply.get(), "")) {
            dismissPrompt(path);
            retire();
            return reply ? unexpectedReply() : false;
        }

        const auto deadline = std::chrono::steady_clock::now() + PromptTimeout;
        while (std::chrono::steady_clock::now() < deadline) {
            while (MessagePtr signal{library.connectionPopMessage(connection)}) {
                const char* signalPath = library.messageGetPath(signal.get());
                const char* signalSender = library.messageGetSender(signal.get());
                if (ownerVanished(signal.get(), sender)) {
                    JST_ERROR("Failed to {} secret: the Secret Service exited during a prompt.", action);
                    return false;
                }
                if (!library.messageIsSignal(signal.get(), SecretPromptInterface, "Completed") ||
                    !signalPath || path != signalPath || !signalSender || sender != signalSender) {
                    continue;
                }
                return promptCompleted(signal.get(), kind, results);
            }
            if (!library.connectionReadWrite(connection, PromptPollMilliseconds)) {
                JST_ERROR("Failed to {} secret: the session bus disconnected during a prompt.", action);
                return false;
            }
        }

        dismissPrompt(path);
        JST_ERROR("Failed to {} secret: the prompt timed out.", action);
        return false;
    }

    bool ownerVanished(DBusMessage* signal, const std::string& sender) {
        const char* signalSender = library.messageGetSender(signal);
        if (!library.messageIsSignal(signal, "org.freedesktop.DBus", "NameOwnerChanged") || !signalSender ||
            std::string(signalSender) != "org.freedesktop.DBus" || !library.messageHasSignature(signal, "sss")) {
            return false;
        }

        DBusMessageIter iter;
        const char* name = nullptr;
        const char* newOwner = nullptr;
        library.iterInit(signal, &iter);
        library.iterGetBasic(&iter, &name);
        library.iterNext(&iter);
        library.iterNext(&iter);
        library.iterGetBasic(&iter, &newOwner);
        return name && newOwner && sender == name && newOwner[0] == '\0';
    }

    bool promptCompleted(DBusMessage* signal, PromptResult kind, std::vector<std::string>& results) {
        DBusMessageIter iter;
        if (!library.messageHasSignature(signal, "bv") || !library.iterInit(signal, &iter)) {
            return unexpectedReply();
        }
        DBusBool dismissed = 0;
        library.iterGetBasic(&iter, &dismissed);
        if (dismissed) {
            JST_ERROR("Failed to {} secret: the prompt was dismissed.", action);
            return false;
        }

        DBusMessageIter variant;
        library.iterNext(&iter);
        library.iterRecurse(&iter, &variant);
        switch (kind) {
            case PromptResult::Any:
                return true;
            case PromptResult::Object: {
                std::string object;
                if (!readObjectPath(&variant, object) || object == SecretNoObject) {
                    return unexpectedReply();
                }
                results.push_back(std::move(object));
                return true;
            }
            case PromptResult::Objects:
                return readObjectPaths(&variant, results) || unexpectedReply();
        }
        return unexpectedReply();
    }

    void closeSession() {
        auto message = call(session.c_str(), "org.freedesktop.Secret.Session", "Close");
        if (!message) {
            retire();
            return;
        }
        ScopedError error(library);
        MessagePtr reply(library.connectionSendWithReplyAndBlock(connection, message.get(),
                                                                 CallTimeoutMilliseconds, error.get()));
        if (!reply || !library.messageHasSignature(reply.get(), "")) {
            retire();
        }
    }

    void dismissPrompt(const std::string& path) {
        auto message = call(path.c_str(), SecretPromptInterface, "Dismiss");
        if (!message) {
            return;
        }
        ScopedError error(library);
        MessagePtr reply(library.connectionSendWithReplyAndBlock(connection, message.get(),
                                                                 CallTimeoutMilliseconds, error.get()));
    }
};

bool ValidBusKey(const Libdbus& library, const char* action, const std::string& service, const std::string& account) {
    if (service.size() + account.size() > SecretMaximumKeyBytes) {
        JST_ERROR("Failed to {} secret: the service and account exceed the D-Bus size limit.", action);
        return false;
    }
    if (!library.validateUtf8(service.c_str(), nullptr) || !library.validateUtf8(account.c_str(), nullptr)) {
        JST_ERROR("Failed to {} secret: the service and account must be valid UTF-8.", action);
        return false;
    }
    return true;
}

}  // namespace

bool SecretStoreAvailable() {
    const auto& library = GetLibdbus();
    if (!library.handle) {
        return false;
    }

    if (InheritedConnection()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(SecretMutex());
    SecretService bus(library, "probe");
    return bus.connect(false) && bus.serviceOwned();
}

Result ReadSecret(const std::string& service, const std::string& account, std::string& value) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    const auto& library = GetLibdbus();
    if (!library.handle || !ValidBusKey(library, "read", service, account)) {
        return Result::ERROR;
    }

    if (RejectInheritedConnection("read")) {
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> lock(SecretMutex());
    SecretService bus(library, "read");
    SecretItems items;
    if (!bus.connect() || !bus.resolveOwner() || !bus.openSession() || !bus.search(service, account, items)) {
        return Result::ERROR;
    }

    std::string item;
    if (!items.unlocked.empty()) {
        item = items.unlocked.front();
    } else if (!items.locked.empty()) {
        item = items.locked.front();
        std::vector<std::string> unlocked;
        if (!bus.unlock({item}, unlocked)) {
            return Result::ERROR;
        }
        if (!ContainsPath(unlocked, item)) {
            JST_ERROR("Failed to read secret: the item is still locked.");
            return Result::ERROR;
        }
    } else {
        return Result::ERROR;
    }

    return bus.getSecret(item, value) ? Result::SUCCESS : Result::ERROR;
}

Result WriteSecret(const std::string& service, const std::string& account, const std::string& value) {
    if (!ValidSecretKey(service, account) || !ValidSecretValue(value)) {
        return Result::ERROR;
    }

    const auto& library = GetLibdbus();
    if (!library.handle || !ValidBusKey(library, "write", service, account)) {
        return Result::ERROR;
    }

    if (value.size() > DBusMaximumArrayBytes) {
        JST_ERROR("Failed to write secret: the value exceeds the D-Bus size limit.");
        return Result::ERROR;
    }

    if (RejectInheritedConnection("write")) {
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> lock(SecretMutex());
    SecretService bus(library, "write");
    const bool stored = bus.connect() &&
                        bus.resolveOwner() &&
                        bus.openSession() &&
                        bus.ensureDefaultCollection() &&
                        bus.unlockDefaultCollection() &&
                        bus.createItem(service, account, value);
    return stored ? Result::SUCCESS : Result::ERROR;
}

Result DeleteSecret(const std::string& service, const std::string& account) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    const auto& library = GetLibdbus();
    if (!library.handle || !ValidBusKey(library, "delete", service, account)) {
        return Result::ERROR;
    }

    if (RejectInheritedConnection("delete")) {
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> lock(SecretMutex());
    SecretService bus(library, "delete");
    SecretItems items;
    if (!bus.connect() || !bus.resolveOwner() || !bus.search(service, account, items)) {
        return Result::ERROR;
    }
    if (items.unlocked.empty() && items.locked.empty()) {
        return Result::SUCCESS;
    }

    bool deleted = true;
    std::vector<std::string> targets = items.unlocked;
    if (!items.locked.empty()) {
        std::vector<std::string> unlocked;
        if (!bus.unlock(items.locked, unlocked)) {
            return Result::ERROR;
        }
        for (const auto& item : items.locked) {
            if (ContainsPath(unlocked, item)) {
                targets.push_back(item);
            }
        }
    }

    for (const auto& item : targets) {
        deleted = bus.deleteItem(item) && deleted;
    }

    SecretItems remaining;
    if (!bus.search(service, account, remaining)) {
        return Result::ERROR;
    }
    if (!remaining.unlocked.empty() || !remaining.locked.empty()) {
        JST_ERROR("Failed to delete secret: the item is still present, it may be locked.");
        return Result::ERROR;
    }

    return deleted ? Result::SUCCESS : Result::ERROR;
}

#else

bool SecretStoreAvailable() {
    return false;
}

Result ReadSecret(const std::string&, const std::string&, std::string&) {
    return Result::ERROR;
}

Result WriteSecret(const std::string&, const std::string&, const std::string&) {
    return Result::ERROR;
}

Result DeleteSecret(const std::string&, const std::string&) {
    return Result::ERROR;
}

#endif

}  // namespace Jetstream::Platform
