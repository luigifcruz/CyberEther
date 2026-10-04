#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_LIVE_MARKDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_LIVE_MARKDOWN_HH

#include <memory>
#include <string>

#include "jetstream/flowgraph.hh"

namespace Jetstream {

class LiveMarkdown {
 public:
    LiveMarkdown();
    explicit LiveMarkdown(std::string source);
    ~LiveMarkdown();

    LiveMarkdown(const LiveMarkdown&) = delete;
    LiveMarkdown& operator=(const LiveMarkdown&) = delete;
    LiveMarkdown(LiveMarkdown&&) noexcept;
    LiveMarkdown& operator=(LiveMarkdown&&) noexcept;

    const std::string& source() const;
    bool dynamic() const;

    std::string expand(const Flowgraph::Environment& environment) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_LIVE_MARKDOWN_HH
