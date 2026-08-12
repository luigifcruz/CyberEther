#ifndef JETSTREAM_RUNTIME_CONTEXT_HH
#define JETSTREAM_RUNTIME_CONTEXT_HH

#include <string>
#include <vector>

#include "jetstream/runtime.hh"

namespace Jetstream {

struct JETSTREAM_API Runtime::Context {
 public:
    struct Diagnostic {
        bool healthy = true;
        std::string status;
        std::vector<std::string> console;
    };

    virtual ~Context() = default;

    virtual Diagnostic diagnostic() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_CONTEXT_HH
