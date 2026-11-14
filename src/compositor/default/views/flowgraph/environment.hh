#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_ENVIRONMENT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_ENVIRONMENT_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphEnvironment {
    struct Config {
        std::string id;
        std::string title;
        std::string search;
        std::string searchHint;
        std::string entryCount;
        Sakura::Table::Nodes nodes;
        Sakura::Callback<void(const std::string&)> onSearchChange;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    Sakura::Component& component();
    const std::string& id() const;
    const std::string& title() const;
    Sakura::Callback<void()> closeCallback() const;

 private:
    Config config;
    Sakura::TextInput searchInput;
    Sakura::Text countText;
    Sakura::Table table;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_ENVIRONMENT_HH
