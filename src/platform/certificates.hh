#ifndef JETSTREAM_PLATFORM_CERTIFICATES_HH
#define JETSTREAM_PLATFORM_CERTIFICATES_HH

#include <string>

namespace Jetstream::Platform::Certificates {

const std::string& Bundle();

template<typename Client>
void Trust(Client& client) {
    if constexpr (requires { client.set_ca_cert_path(std::string()); client.enable_system_ca(true); }) {
        const auto& bundle = Bundle();
        if (bundle.empty()) {
            return;
        }
        client.set_ca_cert_path(bundle);
        client.enable_system_ca(true);
    }
}

}  // namespace Jetstream::Platform::Certificates

#endif  // JETSTREAM_PLATFORM_CERTIFICATES_HH
