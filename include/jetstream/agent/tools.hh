#ifndef JETSTREAM_AGENT_TOOLS_HH
#define JETSTREAM_AGENT_TOOLS_HH

#include "jetstream/agent/provider.hh"
#include "jetstream/block.hh"
#include "jetstream/macros.hh"
#include "jetstream/types.hh"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Jetstream {

class Flowgraph;

}  // namespace Jetstream

namespace Jetstream::Agent {

class JETSTREAM_API Tools {
 public:
    struct Layout {
        struct Node {
            Rect bounds;
            Extent2D<F32> size;
            std::optional<Extent2D<F32>> content;
        };

        struct Viewport {
            Rect visible;
        };

        std::unordered_map<std::string, Node> nodes;
        std::optional<Viewport> viewport;
    };

    class JETSTREAM_API Host {
     public:
        struct DocPage {
            std::string slug;
            std::string title;
            std::string description;
            std::string body;
        };

        struct DocMatch {
            std::string slug;
            std::string title;
            std::string heading;
            std::string anchor;
            std::string snippet;
        };

        virtual ~Host() = default;

        virtual bool has(const std::string& name) = 0;
        virtual Result create(const std::string& name, std::shared_ptr<Flowgraph>& created) = 0;

        virtual std::optional<F32> width(Block::NodeSize size);
        virtual std::optional<std::vector<DocMatch>> searchDocs(const std::string& query, U64 limit);
        virtual std::optional<DocPage> readDocs(const std::string& slug);
    };

    struct Output {
        std::string content;
        std::vector<std::string> images;
        bool error = false;
    };

    struct Flags {
        bool needsGraph = false;
        bool reportsLayout = false;
        bool needsCapture = false;
    };

    struct Definition {
        Provider::Function function;
        Flags flags;
    };

    struct Call {
        std::string tool;
        std::string args;
        const Definition* definition = nullptr;
        std::shared_ptr<Flowgraph> flowgraph;
        std::string create;
        std::optional<Output> output;
    };

    static const std::vector<Definition>& Definitions();
    static std::vector<Provider::Function> Functions(bool captureSupported);
    static Output Error(const std::string& message, bool mayHaveApplied = false);
    static void Annotate(const Call& call, Output& output, const Layout& layout);

    explicit Tools(Host& host);

    Call prepare(const std::shared_ptr<Flowgraph>& target, const std::string& name, const std::string& args) const;
    Output run(Call& call, const Cancellation& cancel = {}) const;
    Output execute(const std::shared_ptr<Flowgraph>& target, const std::string& name, const std::string& args,
                   const Cancellation& cancel = {}) const;

 private:
    Host& host;
};

}  // namespace Jetstream::Agent

#endif  // JETSTREAM_AGENT_TOOLS_HH
