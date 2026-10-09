#ifndef JETSTREAM_TESTS_SUPPORT_ENVIRONMENT_HH
#define JETSTREAM_TESTS_SUPPORT_ENVIRONMENT_HH

#include "jetstream/platform.hh"

#include <optional>
#include <string>
#include <utility>

namespace TestSupport {

class ScopedEnvironment {
 public:
    explicit ScopedEnvironment(std::string name) : variable(std::move(name)) {
        std::string value;
        if (Jetstream::Platform::EnvironmentVariable(variable, value) == Jetstream::Result::SUCCESS) {
            previous = std::move(value);
        }
    }

    ~ScopedEnvironment() {
        (void)Jetstream::Platform::WriteEnvironmentVariable(variable, previous);
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

    bool set(const std::optional<std::string>& value) const {
        return Jetstream::Platform::WriteEnvironmentVariable(variable, value) == Jetstream::Result::SUCCESS;
    }

    const std::string& name() const {
        return variable;
    }

 private:
    std::string variable;
    std::optional<std::string> previous;
};

}  // namespace TestSupport

#endif  // JETSTREAM_TESTS_SUPPORT_ENVIRONMENT_HH
