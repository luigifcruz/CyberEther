#ifndef JETSTREAM_AGENT_SESSION_TRANSCRIPT_HH
#define JETSTREAM_AGENT_SESSION_TRANSCRIPT_HH

#include "jetstream/agent/session.hh"

#include <memory>
#include <span>
#include <string>

namespace Jetstream::Agent::Detail {

// Appends one conversation to a Markdown file as its entries are finalized; a segment that
// is still streaming is checkpointed every few seconds. Every call is safe from any thread,
// never throws, and never touches the session lock. A folder or file that cannot be written
// turns the writer off after a single warning.
class Transcript {
 public:
    struct Origin {
        std::string provider;
        std::string model;
        std::string reasoningEffort;
        // Never written: an error that names the endpoint gets the provider's name instead.
        std::string host;
    };

    explicit Transcript(std::string folder);
    ~Transcript();

    Transcript(const Transcript&) = delete;
    Transcript& operator=(const Transcript&) = delete;

    void user(const Origin& origin, const std::string& text, std::span<const std::string> images = {});
    void stream(Session::Segment::Kind kind, const std::string& chunk);
    void call(const Provider::FunctionCall& call, bool run = true);
    void result(const Provider::FunctionCall& call, const Tools::Output& output);
    void round(const Provider::Response& response, bool summary);
    void event(const std::string& title, const std::string& text);
    void flush();

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Agent::Detail

#endif  // JETSTREAM_AGENT_SESSION_TRANSCRIPT_HH
