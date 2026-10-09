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

struct SecretSchemaAttribute {
    const char* name;
    int type;
};

struct SecretSchema {
    const char* name;
    int flags;
    SecretSchemaAttribute attributes[32];
    int reserved;
    void* reserved1;
    void* reserved2;
    void* reserved3;
    void* reserved4;
    void* reserved5;
    void* reserved6;
    void* reserved7;
};

struct GError {
    unsigned int domain;
    int code;
    char* message;
};

struct GList {
    void* data;
    GList* next;
    GList* prev;
};

constexpr int SecretSearchAll = 1 << 1;

constexpr const char* SecretSchemaName = "ltd.luigi.CyberEther.Secret";
constexpr const char* SecretServiceAttribute = "service";
constexpr const char* SecretAccountAttribute = "account";
constexpr const char* SecretCollectionDefault = "default";

const SecretSchema LibsecretSchema = {
    SecretSchemaName,
    0,
    {
        {SecretServiceAttribute, 0},
        {SecretAccountAttribute, 0},
        {nullptr, 0},
    },
    0,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
};

struct Libsecret {
    void* handle = nullptr;
    int (*store)(const SecretSchema*, const char*, const char*, const char*, void*, GError**, ...) = nullptr;
    char* (*lookup)(const SecretSchema*, void*, GError**, ...) = nullptr;
    int (*clear)(const SecretSchema*, void*, GError**, ...) = nullptr;
    void (*free)(char*) = nullptr;
    GList* (*search)(const SecretSchema*, int, void*, GError**, ...) = nullptr;
    void (*listFreeFull)(GList*, void (*)(void*)) = nullptr;
    void (*objectUnref)(void*) = nullptr;
    void (*errorFree)(GError*) = nullptr;
};

template<typename F>
bool LoadSymbol(void* handle, const char* name, F& target) {
    std::string error;
    target = reinterpret_cast<F>(LoadDynamicLibrarySymbol(handle, name, error));
    return target != nullptr;
}

const Libsecret& GetLibsecret() {
    static Libsecret library;
    static std::once_flag flag;
    std::call_once(flag, [] {
        std::string error;
        library.handle = OpenDynamicLibrary("libsecret-1.so.0", DynamicLibraryVisibility::Local, error);
        if (!library.handle) {
            return;
        }

        const bool loaded = LoadSymbol(library.handle, "secret_password_store_sync", library.store) &&
                            LoadSymbol(library.handle, "secret_password_lookup_sync", library.lookup) &&
                            LoadSymbol(library.handle, "secret_password_clear_sync", library.clear) &&
                            LoadSymbol(library.handle, "secret_password_free", library.free) &&
                            LoadSymbol(library.handle, "secret_password_search_sync", library.search) &&
                            LoadSymbol(library.handle, "g_list_free_full", library.listFreeFull) &&
                            LoadSymbol(library.handle, "g_object_unref", library.objectUnref) &&
                            LoadSymbol(library.handle, "g_error_free", library.errorFree);
        if (!loaded) {
            CloseDynamicLibrary(library.handle);
            library = Libsecret{};
        }
    });
    return library;
}

void ReportSecretError(const char* action, const Libsecret& library, GError* error) {
    if (!error) {
        return;
    }
    JST_ERROR("Failed to {} secret: {} ({}).", action, error->message ? error->message : "unknown error", error->code);
    library.errorFree(error);
}

bool SecretExists(const Libsecret& library, const std::string& service, const std::string& account, GError** error) {
    GList* matches = library.search(&LibsecretSchema, SecretSearchAll, nullptr, error,
                                    SecretServiceAttribute, service.c_str(),
                                    SecretAccountAttribute, account.c_str(),
                                    nullptr);
    if (!matches) {
        return false;
    }
    library.listFreeFull(matches, library.objectUnref);
    return true;
}

}  // namespace

bool SecretStoreAvailable() {
    const auto& library = GetLibsecret();
    if (!library.handle) {
        return false;
    }

    GError* error = nullptr;
    char* probe = library.lookup(&LibsecretSchema, nullptr, &error,
                                 SecretServiceAttribute, "probe",
                                 SecretAccountAttribute, "probe",
                                 nullptr);
    if (probe) {
        library.free(probe);
    }
    if (error) {
        library.errorFree(error);
        return false;
    }
    return true;
}

Result ReadSecret(const std::string& service, const std::string& account, std::string& value) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    const auto& library = GetLibsecret();
    if (!library.handle) {
        return Result::ERROR;
    }

    GError* error = nullptr;
    char* secret = library.lookup(&LibsecretSchema, nullptr, &error,
                                  SecretServiceAttribute, service.c_str(),
                                  SecretAccountAttribute, account.c_str(),
                                  nullptr);
    if (error) {
        ReportSecretError("read", library, error);
        return Result::ERROR;
    }
    if (!secret) {
        return Result::ERROR;
    }

    value = secret;
    library.free(secret);
    return Result::SUCCESS;
}

Result WriteSecret(const std::string& service, const std::string& account, const std::string& value) {
    if (!ValidSecretKey(service, account) || !ValidSecretValue(value)) {
        return Result::ERROR;
    }

    const auto& library = GetLibsecret();
    if (!library.handle) {
        return Result::ERROR;
    }

    const std::string label = service + " (" + account + ")";
    GError* error = nullptr;
    const int stored = library.store(&LibsecretSchema, SecretCollectionDefault, label.c_str(), value.c_str(),
                                     nullptr, &error,
                                     SecretServiceAttribute, service.c_str(),
                                     SecretAccountAttribute, account.c_str(),
                                     nullptr);
    if (error) {
        ReportSecretError("write", library, error);
        return Result::ERROR;
    }

    return stored ? Result::SUCCESS : Result::ERROR;
}

Result DeleteSecret(const std::string& service, const std::string& account) {
    if (!ValidSecretKey(service, account)) {
        return Result::ERROR;
    }

    const auto& library = GetLibsecret();
    if (!library.handle) {
        return Result::ERROR;
    }

    GError* error = nullptr;
    const bool existed = SecretExists(library, service, account, &error);
    if (error) {
        ReportSecretError("delete", library, error);
        return Result::ERROR;
    }
    if (!existed) {
        return Result::SUCCESS;
    }

    library.clear(&LibsecretSchema, nullptr, &error,
                  SecretServiceAttribute, service.c_str(),
                  SecretAccountAttribute, account.c_str(),
                  nullptr);
    if (error) {
        ReportSecretError("delete", library, error);
        return Result::ERROR;
    }

    const bool remaining = SecretExists(library, service, account, &error);
    if (error) {
        ReportSecretError("delete", library, error);
        return Result::ERROR;
    }
    if (remaining) {
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
