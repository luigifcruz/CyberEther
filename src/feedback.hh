#ifndef JETSTREAM_FEEDBACK_HH
#define JETSTREAM_FEEDBACK_HH

#include "jetstream/types.hh"

#include <memory>
#include <string>

namespace Jetstream {

class Feedback {
 public:
    struct Snapshot {
        enum class Status {
            Idle,
            Submitting,
            Success,
            Error,
        };

        Status status = Status::Idle;
        std::string message;
        std::string id;
    };

    static constexpr U64 MIN_LENGTH = 10;
    static constexpr U64 MAX_LENGTH = 4000;
    static constexpr const char* ENDPOINT = "https://cyberether.org/api/v1/feedback";

    Feedback();
    ~Feedback();

    Feedback(const Feedback&) = delete;
    Feedback& operator=(const Feedback&) = delete;

    void submit(const std::string& text);
    Snapshot snapshot() const;
    void clear();
    void shutdown();

 private:
    struct Impl;
    friend struct FetchContext;
    std::shared_ptr<Impl> pimpl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_FEEDBACK_HH
