#include "certificates.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#if defined(JST_OS_LINUX) && defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
#include <openssl/bio.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/x509.h>

#include <filesystem>
#include <vector>
#endif

namespace Jetstream::Platform::Certificates {

#if defined(JST_OS_LINUX) && defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)

namespace {

constexpr const char* kBundles[] = {
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/var/lib/ca-certificates/ca-bundle.pem",
    "/etc/ca-certificates/extracted/tls-ca-bundle.pem",
    "/etc/ssl/certs/ca-bundle.crt",
    "/etc/ssl/cert.pem",
    "/etc/pki/tls/cert.pem",
    "/etc/pki/tls/cacert.pem",
};

int RefusePassphrase(char*, int, int, void*) {
    return -1;
}

bool Loadable(const std::string& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        return false;
    }

    BIO* file = BIO_new_file(path.c_str(), "r");
    if (!file) {
        ERR_clear_error();
        return false;
    }

    STACK_OF(X509_INFO)* entries = PEM_X509_INFO_read_bio(file, nullptr, RefusePassphrase, nullptr);
    BIO_free(file);

    bool found = false;
    if (entries) {
        for (int i = 0; i < sk_X509_INFO_num(entries) && !found; ++i) {
            found = sk_X509_INFO_value(entries, i)->x509 != nullptr;
        }
        sk_X509_INFO_pop_free(entries, X509_INFO_free);
    }

    ERR_clear_error();
    return found;
}

bool Environment(const char* name, std::string& value) {
    return OPENSSL_issetugid() == 0 && EnvironmentVariable(name, value) == Result::SUCCESS;
}

bool Overridden(const char* name) {
    std::string value;
    return Environment(name, value);
}

std::string Resolve() {
    std::vector<std::string> candidates;

    std::string nix;
    if (Environment("NIX_SSL_CERT_FILE", nix) && std::filesystem::path(nix).is_absolute()) {
        candidates.push_back(std::move(nix));
    }
    candidates.insert(candidates.end(), std::begin(kBundles), std::end(kBundles));

    for (const auto& candidate : candidates) {
        if (Loadable(candidate)) {
            JST_DEBUG("[PLATFORM] Using the CA bundle at '{}'.", candidate);
            return candidate;
        }
    }

    JST_WARN("[PLATFORM] No system CA bundle was found. HTTPS will only trust the OpenSSL defaults.");
    return {};
}

}  // namespace

const std::string& Bundle() {
    static const std::string none;
    if (Overridden("SSL_CERT_FILE") || Overridden("SSL_CERT_DIR")) {
        return none;
    }

    static const std::string bundle = Resolve();
    return bundle;
}

#else

const std::string& Bundle() {
    static const std::string bundle;
    return bundle;
}

#endif

}  // namespace Jetstream::Platform::Certificates
