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
#include <chrono>
#include <climits>
#include <cstdint>
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

// Linux talks to the freedesktop Secret Service over D-Bus through libdbus,
// loaded at runtime. libdbus does not depend on GLib, so it shares the process
// with the bundled GLib without a separate link namespace. A namespace brings a
// second libc whose thread-specific keys collide with the primary libc's and
// crash threads on exit. Items carry the attributes libsecret writes for the
// same schema, so secrets stored by earlier releases stay readable.

namespace {

struct DBusConnection;
struct DBusMessage;

// Mirrors of the public libdbus structures, which are ABI stable.

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
constexpr int DBusTypeVariant = 'v';

constexpr const char* SecretsName = "org.freedesktop.secrets";
constexpr const char* SecretsPath = "/org/freedesktop/secrets";
constexpr const char* ServiceInterface = "org.freedesktop.Secret.Service";
constexpr const char* CollectionInterface = "org.freedesktop.Secret.Collection";
constexpr const char* ItemInterface = "org.freedesktop.Secret.Item";
constexpr const char* PromptInterface = "org.freedesktop.Secret.Prompt";
constexpr const char* NoPrompt = "/";

constexpr const char* SecretSchemaName = "ltd.luigi.CyberEther.Secret";
constexpr const char* SecretSchemaAttribute = "xdg:schema";
constexpr const char* SecretServiceAttribute = "service";
constexpr const char* SecretAccountAttribute = "account";
constexpr const char* SecretCollectionDefault = "default";
constexpr const char* SecretCollectionLabel = "Default keyring";
constexpr const char* SecretContentType = "text/plain";
constexpr const char* OutOfMemory = "out of memory";

constexpr int CallTimeoutMilliseconds = 25000;
constexpr int PromptPollMilliseconds = 250;
constexpr auto PromptTimeout = std::chrono::minutes(2);

using Attributes = std::vector<std::pair<std::string, std::string>>;

struct Libdbus {
    void* handle = nullptr;
    DBusBool (*threadsInitDefault)() = nullptr;
    void (*errorInit)(DBusError*) = nullptr;
    void (*errorFree)(DBusError*) = nullptr;
    DBusBool (*errorIsSet)(const DBusError*) = nullptr;
    void (*connectionSetChangeSigpipe)(DBusBool) = nullptr;
    DBusConnection* (*busGetPrivate)(int, DBusError*) = nullptr;
    void (*busAddMatch)(DBusConnection*, const char*, DBusError*) = nullptr;
    void (*busRemoveMatch)(DBusConnection*, const char*, DBusError*) = nullptr;
    void (*connectionSetExitOnDisconnect)(DBusConnection*, DBusBool) = nullptr;
    void (*connectionClose)(DBusConnection*) = nullptr;
    void (*connectionUnref)(DBusConnection*) = nullptr;
    DBusMessage* (*connectionSendWithReplyAndBlock)(DBusConnection*, DBusMessage*, int, DBusError*) = nullptr;
    DBusBool (*connectionReadWrite)(DBusConnection*, int) = nullptr;
    DBusMessage* (*connectionPopMessage)(DBusConnection*) = nullptr;
    DBusMessage* (*messageNewMethodCall)(const char*, const char*, const char*, const char*) = nullptr;
    void (*messageUnref)(DBusMessage*) = nullptr;
    DBusBool (*messageIsSignal)(DBusMessage*, const char*, const char*) = nullptr;
    DBusBool (*messageHasPath)(DBusMessage*, const char*) = nullptr;
    void (*iterInitAppend)(DBusMessage*, DBusMessageIter*) = nullptr;
    DBusBool (*iterAppendBasic)(DBusMessageIter*, int, const void*) = nullptr;
    DBusBool (*iterAppendFixedArray)(DBusMessageIter*, int, const void*, int) = nullptr;
    DBusBool (*iterOpenContainer)(DBusMessageIter*, int, const char*, DBusMessageIter*) = nullptr;
    DBusBool (*iterCloseContainer)(DBusMessageIter*, DBusMessageIter*) = nullptr;
    DBusBool (*iterInit)(DBusMessage*, DBusMessageIter*) = nullptr;
    int (*iterGetArgType)(DBusMessageIter*) = nullptr;
    void (*iterGetBasic)(DBusMessageIter*, void*) = nullptr;
    void (*iterRecurse)(DBusMessageIter*, DBusMessageIter*) = nullptr;
    DBusBool (*iterNext)(DBusMessageIter*) = nullptr;
    void (*iterGetFixedArray)(DBusMessageIter*, void*, int*) = nullptr;
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
            JST_DEBUG("Secret storage is unavailable: {}.", error);
            return;
        }

        void* handle = library.handle;
        const bool loaded =
            LoadSymbol(handle, "dbus_threads_init_default", library.threadsInitDefault) &&
            LoadSymbol(handle, "dbus_error_init", library.errorInit) &&
            LoadSymbol(handle, "dbus_error_free", library.errorFree) &&
            LoadSymbol(handle, "dbus_error_is_set", library.errorIsSet) &&
            LoadSymbol(handle, "dbus_connection_set_change_sigpipe", library.connectionSetChangeSigpipe) &&
            LoadSymbol(handle, "dbus_bus_get_private", library.busGetPrivate) &&
            LoadSymbol(handle, "dbus_bus_add_match", library.busAddMatch) &&
            LoadSymbol(handle, "dbus_bus_remove_match", library.busRemoveMatch) &&
            LoadSymbol(handle, "dbus_connection_set_exit_on_disconnect", library.connectionSetExitOnDisconnect) &&
            LoadSymbol(handle, "dbus_connection_close", library.connectionClose) &&
            LoadSymbol(handle, "dbus_connection_unref", library.connectionUnref) &&
            LoadSymbol(handle, "dbus_connection_send_with_reply_and_block", library.connectionSendWithReplyAndBlock) &&
            LoadSymbol(handle, "dbus_connection_read_write", library.connectionReadWrite) &&
            LoadSymbol(handle, "dbus_connection_pop_message", library.connectionPopMessage) &&
            LoadSymbol(handle, "dbus_message_new_method_call", library.messageNewMethodCall) &&
            LoadSymbol(handle, "dbus_message_unref", library.messageUnref) &&
            LoadSymbol(handle, "dbus_message_is_signal", library.messageIsSignal) &&
            LoadSymbol(handle, "dbus_message_has_path", library.messageHasPath) &&
            LoadSymbol(handle, "dbus_message_iter_init_append", library.iterInitAppend) &&
            LoadSymbol(handle, "dbus_message_iter_append_basic", library.iterAppendBasic) &&
            LoadSymbol(handle, "dbus_message_iter_append_fixed_array", library.iterAppendFixedArray) &&
            LoadSymbol(handle, "dbus_message_iter_open_container", library.iterOpenContainer) &&
            LoadSymbol(handle, "dbus_message_iter_close_container", library.iterCloseContainer) &&
            LoadSymbol(handle, "dbus_message_iter_init", library.iterInit) &&
            LoadSymbol(handle, "dbus_message_iter_get_arg_type", library.iterGetArgType) &&
            LoadSymbol(handle, "dbus_message_iter_get_basic", library.iterGetBasic) &&
            LoadSymbol(handle, "dbus_message_iter_recurse", library.iterRecurse) &&
            LoadSymbol(handle, "dbus_message_iter_next", library.iterNext) &&
            LoadSymbol(handle, "dbus_message_iter_get_fixed_array", library.iterGetFixedArray);
        if (!loaded || !library.threadsInitDefault()) {
            JST_DEBUG("Secret storage is unavailable: libdbus is missing required symbols.");
            CloseDynamicLibrary(library.handle);
            library = Libdbus{};
            return;
        }

        // Keep the process SIGPIPE disposition; libdbus writes with MSG_NOSIGNAL.
        library.connectionSetChangeSigpipe(0);
    });
    return library;
}

// libdbus aborts the process on invalid UTF-8, so strings are validated first.
// This follows the strictest libdbus rules, which also reject noncharacters.
bool ValidUtf8(const std::string& text) {
    static constexpr std::uint32_t Minimum[] = {0, 0, 0x80, 0x800, 0x10000};

    std::size_t index = 0;
    while (index < text.size()) {
        const auto lead = static_cast<unsigned char>(text[index]);
        std::size_t length = 0;
        std::uint32_t codepoint = 0;
        if (lead < 0x80) {
            index += 1;
            continue;
        } else if ((lead & 0xE0) == 0xC0) {
            length = 2;
            codepoint = lead & 0x1F;
        } else if ((lead & 0xF0) == 0xE0) {
            length = 3;
            codepoint = lead & 0x0F;
        } else if ((lead & 0xF8) == 0xF0) {
            length = 4;
            codepoint = lead & 0x07;
        } else {
            return false;
        }

        if (index + length > text.size()) {
            return false;
        }
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto next = static_cast<unsigned char>(text[index + offset]);
            if ((next & 0xC0) != 0x80) {
                return false;
            }
            codepoint = (codepoint << 6) | (next & 0x3F);
        }

        if (codepoint < Minimum[length] || codepoint > 0x10FFFF ||
            (codepoint >= 0xD800 && codepoint <= 0xDFFF) ||
            (codepoint >= 0xFDD0 && codepoint <= 0xFDEF) ||
            (codepoint & 0xFFFE) == 0xFFFE) {
            return false;
        }
        index += length;
    }
    return true;
}

bool ValidLinuxSecretKey(const std::string& service, const std::string& account) {
    return ValidSecretKey(service, account) && ValidUtf8(service) && ValidUtf8(account);
}

Attributes SecretAttributes(const std::string& service, const std::string& account) {
    return {
        {SecretSchemaAttribute, SecretSchemaName},
        {SecretServiceAttribute, service},
        {SecretAccountAttribute, account},
    };
}

class Message {
 public:
    Message(const Libdbus& dbus, DBusMessage* message) : dbus(dbus), message(message) {}
    ~Message() {
        if (message) {
            dbus.messageUnref(message);
        }
    }

    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;

    DBusMessage* get() const {
        return message;
    }

    explicit operator bool() const {
        return message != nullptr;
    }

 private:
    const Libdbus& dbus;
    DBusMessage* message;
};

bool AppendString(const Libdbus& dbus, DBusMessageIter* iter, int type, const std::string& text) {
    const char* value = text.c_str();
    return dbus.iterAppendBasic(iter, type, &value) != 0;
}

bool AppendObjectPaths(const Libdbus& dbus, DBusMessageIter* iter, const std::vector<std::string>& paths) {
    DBusMessageIter array;
    if (!dbus.iterOpenContainer(iter, DBusTypeArray, "o", &array)) {
        return false;
    }
    for (const auto& path : paths) {
        if (!AppendString(dbus, &array, DBusTypeObjectPath, path)) {
            return false;
        }
    }
    return dbus.iterCloseContainer(iter, &array) != 0;
}

bool AppendAttributes(const Libdbus& dbus, DBusMessageIter* iter, const Attributes& attributes) {
    DBusMessageIter array;
    if (!dbus.iterOpenContainer(iter, DBusTypeArray, "{ss}", &array)) {
        return false;
    }
    for (const auto& [key, value] : attributes) {
        DBusMessageIter entry;
        if (!dbus.iterOpenContainer(&array, DBusTypeDictEntry, nullptr, &entry) ||
            !AppendString(dbus, &entry, DBusTypeString, key) ||
            !AppendString(dbus, &entry, DBusTypeString, value) ||
            !dbus.iterCloseContainer(&array, &entry)) {
            return false;
        }
    }
    return dbus.iterCloseContainer(iter, &array) != 0;
}

template<typename Content>
bool AppendProperty(const Libdbus& dbus,
                    DBusMessageIter* array,
                    const std::string& key,
                    const char* signature,
                    Content&& content) {
    DBusMessageIter entry;
    DBusMessageIter variant;
    return dbus.iterOpenContainer(array, DBusTypeDictEntry, nullptr, &entry) &&
           AppendString(dbus, &entry, DBusTypeString, key) &&
           dbus.iterOpenContainer(&entry, DBusTypeVariant, signature, &variant) &&
           content(&variant) &&
           dbus.iterCloseContainer(&entry, &variant) &&
           dbus.iterCloseContainer(array, &entry);
}

bool ReadObjectPath(const Libdbus& dbus, DBusMessageIter* iter, std::string& path) {
    if (dbus.iterGetArgType(iter) != DBusTypeObjectPath) {
        return false;
    }
    const char* value = nullptr;
    dbus.iterGetBasic(iter, &value);
    path = value ? value : "";
    return true;
}

// Reads either a single object path or an array of them.
bool ReadObjectPaths(const Libdbus& dbus, DBusMessageIter* iter, std::vector<std::string>& paths) {
    std::string path;
    if (ReadObjectPath(dbus, iter, path)) {
        paths.push_back(std::move(path));
        return true;
    }
    if (dbus.iterGetArgType(iter) != DBusTypeArray) {
        return false;
    }

    DBusMessageIter array;
    dbus.iterRecurse(iter, &array);
    while (ReadObjectPath(dbus, &array, path)) {
        paths.push_back(std::move(path));
        dbus.iterNext(&array);
    }
    return true;
}

// One private bus connection and one plain Secret Service session per
// operation. The session bus is private to the user, the same trust boundary
// libsecret falls back to when it can't negotiate transport encryption.
class SecretSession {
 public:
    explicit SecretSession(const Libdbus& dbus) : dbus(dbus) {}

    ~SecretSession() {
        if (connection) {
            dbus.connectionClose(connection);
            dbus.connectionUnref(connection);
        }
    }

    SecretSession(const SecretSession&) = delete;
    SecretSession& operator=(const SecretSession&) = delete;

    const std::string& error() const {
        return failure;
    }

    bool open() {
        DBusError error;
        dbus.errorInit(&error);
        connection = dbus.busGetPrivate(DBusBusSession, &error);
        if (!connection) {
            failure = dbus.errorIsSet(&error) && error.message ? error.message : "can't connect to the session bus";
            dbus.errorFree(&error);
            return false;
        }
        dbus.errorFree(&error);
        dbus.connectionSetExitOnDisconnect(connection, 0);

        Message request(dbus, method(SecretsPath, ServiceInterface, "OpenSession"));
        if (!request) {
            return fail(OutOfMemory);
        }
        DBusMessageIter args;
        DBusMessageIter input;
        dbus.iterInitAppend(request.get(), &args);
        if (!AppendString(dbus, &args, DBusTypeString, "plain") ||
            !dbus.iterOpenContainer(&args, DBusTypeVariant, "s", &input) ||
            !AppendString(dbus, &input, DBusTypeString, "") ||
            !dbus.iterCloseContainer(&args, &input)) {
            return fail(OutOfMemory);
        }

        Message reply(dbus, call(request.get()));
        if (!reply) {
            return false;
        }
        DBusMessageIter result;
        if (!dbus.iterInit(reply.get(), &result) || !dbus.iterNext(&result) ||
            !ReadObjectPath(dbus, &result, session)) {
            return fail("unexpected reply to OpenSession");
        }
        return true;
    }

    // Finds matching items, unlocking any that are locked.
    bool search(const Attributes& attributes, std::vector<std::string>& items) {
        Message request(dbus, method(SecretsPath, ServiceInterface, "SearchItems"));
        if (!request) {
            return fail(OutOfMemory);
        }
        DBusMessageIter args;
        dbus.iterInitAppend(request.get(), &args);
        if (!AppendAttributes(dbus, &args, attributes)) {
            return fail(OutOfMemory);
        }

        Message reply(dbus, call(request.get()));
        if (!reply) {
            return false;
        }
        std::vector<std::string> locked;
        DBusMessageIter result;
        if (!dbus.iterInit(reply.get(), &result) || !ReadObjectPaths(dbus, &result, items) ||
            !dbus.iterNext(&result) || !ReadObjectPaths(dbus, &result, locked)) {
            return fail("unexpected reply to SearchItems");
        }

        if (!locked.empty()) {
            if (!unlock(locked)) {
                return false;
            }
            items.insert(items.end(), locked.begin(), locked.end());
        }
        return true;
    }

    bool secret(const std::string& item, std::string& value) {
        Message request(dbus, method(item.c_str(), ItemInterface, "GetSecret"));
        if (!request) {
            return fail(OutOfMemory);
        }
        DBusMessageIter args;
        dbus.iterInitAppend(request.get(), &args);
        if (!AppendString(dbus, &args, DBusTypeObjectPath, session)) {
            return fail(OutOfMemory);
        }

        Message reply(dbus, call(request.get()));
        if (!reply) {
            return false;
        }

        // Reply is (oayays): session, parameters, value, content type.
        DBusMessageIter result;
        DBusMessageIter fields;
        DBusMessageIter bytes;
        if (!dbus.iterInit(reply.get(), &result) || dbus.iterGetArgType(&result) != DBusTypeStruct) {
            return fail("unexpected reply to GetSecret");
        }
        dbus.iterRecurse(&result, &fields);
        if (!dbus.iterNext(&fields) || !dbus.iterNext(&fields) || dbus.iterGetArgType(&fields) != DBusTypeArray) {
            return fail("unexpected reply to GetSecret");
        }
        dbus.iterRecurse(&fields, &bytes);
        const int type = dbus.iterGetArgType(&bytes);
        if (type != DBusTypeByte && type != DBusTypeInvalid) {
            return fail("unexpected reply to GetSecret");
        }

        const char* data = nullptr;
        int size = 0;
        dbus.iterGetFixedArray(&bytes, &data, &size);
        value.assign(data ? data : "", data && size > 0 ? static_cast<std::size_t>(size) : 0);
        return true;
    }

    bool store(const std::string& label, const Attributes& attributes, const std::string& value) {
        std::string target;
        if (!defaultCollection(target)) {
            return false;
        }

        Message request(dbus, method(target.c_str(), CollectionInterface, "CreateItem"));
        if (!request) {
            return fail(OutOfMemory);
        }

        DBusMessageIter args;
        DBusMessageIter properties;
        DBusMessageIter secret;
        DBusMessageIter parameters;
        DBusMessageIter bytes;
        const auto* data = reinterpret_cast<const unsigned char*>(value.data());
        const DBusBool replace = 1;
        dbus.iterInitAppend(request.get(), &args);
        const bool built =
            dbus.iterOpenContainer(&args, DBusTypeArray, "{sv}", &properties) &&
            AppendProperty(dbus, &properties, "org.freedesktop.Secret.Item.Label", "s", [&](DBusMessageIter* iter) {
                return AppendString(dbus, iter, DBusTypeString, label);
            }) &&
            AppendProperty(dbus, &properties, "org.freedesktop.Secret.Item.Attributes", "a{ss}",
                           [&](DBusMessageIter* iter) {
                return AppendAttributes(dbus, iter, attributes);
            }) &&
            dbus.iterCloseContainer(&args, &properties) &&
            dbus.iterOpenContainer(&args, DBusTypeStruct, nullptr, &secret) &&
            AppendString(dbus, &secret, DBusTypeObjectPath, session) &&
            dbus.iterOpenContainer(&secret, DBusTypeArray, "y", &parameters) &&
            dbus.iterCloseContainer(&secret, &parameters) &&
            dbus.iterOpenContainer(&secret, DBusTypeArray, "y", &bytes) &&
            dbus.iterAppendFixedArray(&bytes, DBusTypeByte, &data, static_cast<int>(value.size())) &&
            dbus.iterCloseContainer(&secret, &bytes) &&
            AppendString(dbus, &secret, DBusTypeString, SecretContentType) &&
            dbus.iterCloseContainer(&args, &secret) &&
            dbus.iterAppendBasic(&args, DBusTypeBoolean, &replace);
        if (!built) {
            return fail(OutOfMemory);
        }

        Message reply(dbus, call(request.get()));
        if (!reply) {
            return false;
        }
        std::string item;
        std::string prompt;
        DBusMessageIter result;
        if (!dbus.iterInit(reply.get(), &result) || !ReadObjectPath(dbus, &result, item) ||
            !dbus.iterNext(&result) || !ReadObjectPath(dbus, &result, prompt)) {
            return fail("unexpected reply to CreateItem");
        }

        std::vector<std::string> created;
        return complete(prompt, created);
    }

    bool remove(const std::string& item) {
        Message request(dbus, method(item.c_str(), ItemInterface, "Delete"));
        if (!request) {
            return fail(OutOfMemory);
        }

        Message reply(dbus, call(request.get()));
        if (!reply) {
            return false;
        }
        std::string prompt;
        DBusMessageIter result;
        if (!dbus.iterInit(reply.get(), &result) || !ReadObjectPath(dbus, &result, prompt)) {
            return fail("unexpected reply to Delete");
        }

        std::vector<std::string> ignored;
        return complete(prompt, ignored);
    }

 private:
    const Libdbus& dbus;
    DBusConnection* connection = nullptr;
    std::string session;
    std::string failure;

    bool fail(std::string message) {
        failure = std::move(message);
        return false;
    }

    DBusMessage* method(const char* path, const char* interface, const char* member) {
        return dbus.messageNewMethodCall(SecretsName, path, interface, member);
    }

    DBusMessage* call(DBusMessage* request) {
        DBusError error;
        dbus.errorInit(&error);
        DBusMessage* reply = dbus.connectionSendWithReplyAndBlock(connection, request, CallTimeoutMilliseconds, &error);
        if (!reply) {
            failure = dbus.errorIsSet(&error) && error.message ? error.message : "no reply from the secret service";
        }
        dbus.errorFree(&error);
        return reply;
    }

    bool unlock(std::vector<std::string>& objects) {
        Message request(dbus, method(SecretsPath, ServiceInterface, "Unlock"));
        if (!request) {
            return fail(OutOfMemory);
        }
        DBusMessageIter args;
        dbus.iterInitAppend(request.get(), &args);
        if (!AppendObjectPaths(dbus, &args, objects)) {
            return fail(OutOfMemory);
        }

        Message reply(dbus, call(request.get()));
        if (!reply) {
            return false;
        }
        std::vector<std::string> unlocked;
        std::string prompt;
        DBusMessageIter result;
        if (!dbus.iterInit(reply.get(), &result) || !ReadObjectPaths(dbus, &result, unlocked) ||
            !dbus.iterNext(&result) || !ReadObjectPath(dbus, &result, prompt)) {
            return fail("unexpected reply to Unlock");
        }
        if (!complete(prompt, unlocked)) {
            return false;
        }
        if (unlocked.empty()) {
            return fail("the keyring is locked");
        }

        objects = std::move(unlocked);
        return true;
    }

    // Resolves the default collection, creating it when missing, and unlocks it.
    bool defaultCollection(std::string& path) {
        {
            Message request(dbus, method(SecretsPath, ServiceInterface, "ReadAlias"));
            if (!request) {
                return fail(OutOfMemory);
            }
            DBusMessageIter args;
            dbus.iterInitAppend(request.get(), &args);
            if (!AppendString(dbus, &args, DBusTypeString, SecretCollectionDefault)) {
                return fail(OutOfMemory);
            }

            Message reply(dbus, call(request.get()));
            if (!reply) {
                return false;
            }
            DBusMessageIter result;
            if (!dbus.iterInit(reply.get(), &result) || !ReadObjectPath(dbus, &result, path)) {
                return fail("unexpected reply to ReadAlias");
            }
        }

        if (path == NoPrompt) {
            Message request(dbus, method(SecretsPath, ServiceInterface, "CreateCollection"));
            if (!request) {
                return fail(OutOfMemory);
            }
            DBusMessageIter args;
            DBusMessageIter properties;
            dbus.iterInitAppend(request.get(), &args);
            const bool built =
                dbus.iterOpenContainer(&args, DBusTypeArray, "{sv}", &properties) &&
                AppendProperty(dbus, &properties, "org.freedesktop.Secret.Collection.Label", "s",
                               [&](DBusMessageIter* iter) {
                    return AppendString(dbus, iter, DBusTypeString, SecretCollectionLabel);
                }) &&
                dbus.iterCloseContainer(&args, &properties) &&
                AppendString(dbus, &args, DBusTypeString, SecretCollectionDefault);
            if (!built) {
                return fail(OutOfMemory);
            }

            Message reply(dbus, call(request.get()));
            if (!reply) {
                return false;
            }
            std::string prompt;
            DBusMessageIter result;
            if (!dbus.iterInit(reply.get(), &result) || !ReadObjectPath(dbus, &result, path) ||
                !dbus.iterNext(&result) || !ReadObjectPath(dbus, &result, prompt)) {
                return fail("unexpected reply to CreateCollection");
            }

            std::vector<std::string> created;
            if (!complete(prompt, created)) {
                return false;
            }
            if (path == NoPrompt && !created.empty()) {
                path = created.front();
            }
            if (path == NoPrompt) {
                return fail("can't create the default keyring");
            }
        }

        std::vector<std::string> collection = {path};
        return unlock(collection);
    }

    // Runs a prompt when the service asks for one. The prompt's result object
    // paths are appended to `objects`.
    bool complete(const std::string& prompt, std::vector<std::string>& objects) {
        if (prompt == NoPrompt) {
            return true;
        }

        const std::string rule = std::string("type='signal',interface='") + PromptInterface +
                                 "',member='Completed',path='" + prompt + "'";
        DBusError error;
        dbus.errorInit(&error);
        dbus.busAddMatch(connection, rule.c_str(), &error);
        if (dbus.errorIsSet(&error)) {
            failure = error.message ? error.message : "can't watch the keyring prompt";
            dbus.errorFree(&error);
            return false;
        }

        const bool completed = await(prompt, objects);
        dbus.busRemoveMatch(connection, rule.c_str(), nullptr);
        return completed;
    }

    bool await(const std::string& prompt, std::vector<std::string>& objects) {
        {
            Message request(dbus, method(prompt.c_str(), PromptInterface, "Prompt"));
            if (!request) {
                return fail(OutOfMemory);
            }
            DBusMessageIter args;
            dbus.iterInitAppend(request.get(), &args);
            if (!AppendString(dbus, &args, DBusTypeString, "")) {
                return fail(OutOfMemory);
            }
            Message reply(dbus, call(request.get()));
            if (!reply) {
                return false;
            }
        }

        const auto deadline = std::chrono::steady_clock::now() + PromptTimeout;
        while (std::chrono::steady_clock::now() < deadline) {
            Message message(dbus, dbus.connectionPopMessage(connection));
            if (!message) {
                if (!dbus.connectionReadWrite(connection, PromptPollMilliseconds)) {
                    return fail("lost the session bus connection");
                }
                continue;
            }
            if (!dbus.messageIsSignal(message.get(), PromptInterface, "Completed") ||
                !dbus.messageHasPath(message.get(), prompt.c_str())) {
                continue;
            }

            // Signal is (bv): dismissed, result.
            DBusMessageIter result;
            DBusBool dismissed = 1;
            if (dbus.iterInit(message.get(), &result) && dbus.iterGetArgType(&result) == DBusTypeBoolean) {
                dbus.iterGetBasic(&result, &dismissed);
                if (dbus.iterNext(&result) && dbus.iterGetArgType(&result) == DBusTypeVariant) {
                    DBusMessageIter variant;
                    dbus.iterRecurse(&result, &variant);
                    ReadObjectPaths(dbus, &variant, objects);
                }
            }
            return dismissed ? fail("the keyring prompt was dismissed") : true;
        }
        return fail("timed out waiting for the keyring prompt");
    }
};

void ReportSecretError(const char* action, const SecretSession& session) {
    JST_ERROR("Failed to {} secret: {}.", action, session.error());
}

}  // namespace

bool SecretStoreAvailable() {
    const auto& dbus = GetLibdbus();
    if (!dbus.handle) {
        return false;
    }

    SecretSession session(dbus);
    return session.open();
}

Result ReadSecret(const std::string& service, const std::string& account, std::string& value) {
    if (!ValidLinuxSecretKey(service, account)) {
        return Result::ERROR;
    }

    const auto& dbus = GetLibdbus();
    if (!dbus.handle) {
        return Result::ERROR;
    }

    SecretSession session(dbus);
    std::vector<std::string> items;
    if (!session.open() || !session.search(SecretAttributes(service, account), items)) {
        ReportSecretError("read", session);
        return Result::ERROR;
    }
    if (items.empty()) {
        return Result::ERROR;
    }

    std::string secret;
    if (!session.secret(items.front(), secret)) {
        ReportSecretError("read", session);
        return Result::ERROR;
    }

    value = std::move(secret);
    return Result::SUCCESS;
}

Result WriteSecret(const std::string& service, const std::string& account, const std::string& value) {
    if (!ValidLinuxSecretKey(service, account) || !ValidSecretValue(value) ||
        value.size() > static_cast<std::size_t>(INT_MAX)) {
        return Result::ERROR;
    }

    const auto& dbus = GetLibdbus();
    if (!dbus.handle) {
        return Result::ERROR;
    }

    SecretSession session(dbus);
    const std::string label = service + " (" + account + ")";
    if (!session.open() || !session.store(label, SecretAttributes(service, account), value)) {
        ReportSecretError("write", session);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DeleteSecret(const std::string& service, const std::string& account) {
    if (!ValidLinuxSecretKey(service, account)) {
        return Result::ERROR;
    }

    const auto& dbus = GetLibdbus();
    if (!dbus.handle) {
        return Result::ERROR;
    }

    SecretSession session(dbus);
    const auto attributes = SecretAttributes(service, account);
    std::vector<std::string> items;
    if (!session.open() || !session.search(attributes, items)) {
        ReportSecretError("delete", session);
        return Result::ERROR;
    }
    if (items.empty()) {
        return Result::SUCCESS;
    }

    for (const auto& item : items) {
        if (!session.remove(item)) {
            ReportSecretError("delete", session);
            return Result::ERROR;
        }
    }

    std::vector<std::string> remaining;
    if (!session.search(attributes, remaining)) {
        ReportSecretError("delete", session);
        return Result::ERROR;
    }
    if (!remaining.empty()) {
        JST_ERROR("Failed to delete secret: the item is still present, it may be locked.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
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
