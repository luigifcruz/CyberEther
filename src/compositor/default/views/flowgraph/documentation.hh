#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphDocumentation {
    struct Config {
        std::string id;
        std::string title;
        std::string markdown;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    Sakura::Component& component();
    const std::string& id() const;
    const std::string& title() const;
    Sakura::Callback<void()> closeCallback() const;

 private:
    Config config;
    Sakura::Markdown markdown;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH
