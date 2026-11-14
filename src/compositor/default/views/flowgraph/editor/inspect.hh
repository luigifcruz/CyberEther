#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_INSPECT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_INSPECT_HH

#include "jetstream/parser.hh"
#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphNodeInspector {
    struct Config {
        std::string id;
        std::string name;
        Parser::Map value;
        Sakura::Callback<void(Parser::Map)> onApply;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    void open();
    Sakura::FloatingLayer::Window window();

 private:
    bool canApply() const;
    void apply();

    Config config;
    std::string buffer;
    std::string appliedBuffer;
    std::string status;
    Sakura::CodeEditor::StatusTone statusTone =
        Sakura::CodeEditor::StatusTone::Info;
    Sakura::CodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_INSPECT_HH
