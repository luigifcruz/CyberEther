#include "executor.hh"

#include "../../model/callbacks.hh"
#include "../../model/meta.hh"
#include "../../model/state.hh"
#include "../docs.hh"
#include "jetstream/agent/tools.hh"
#include "jetstream/instance.hh"

#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>

namespace Jetstream {

namespace {

constexpr U64 kSettledFrames = 3;
constexpr U64 kMaxSettleFrames = 60;

bool Near(F32 a, F32 b) {
    return std::abs(a - b) <= 0.5f;
}

bool Near(const Extent2D<F32>& a, const Extent2D<F32>& b) {
    return Near(a.x, b.x) && Near(a.y, b.y);
}

bool Near(const Rect& a, const Rect& b) {
    return Near(a.x, b.x) && Near(a.y, b.y) && Near(a.width, b.width) && Near(a.height, b.height);
}

bool SameNode(const Agent::Tools::Layout::Node& a, const Agent::Tools::Layout::Node& b) {
    if (!Near(a.bounds, b.bounds) || !Near(a.size, b.size)) {
        return false;
    }
    if (a.content.has_value() != b.content.has_value()) {
        return false;
    }
    if (a.content.has_value() && !Near(*a.content, *b.content)) {
        return false;
    }
    return true;
}

bool SameNodes(const Agent::Tools::Layout& a, const Agent::Tools::Layout& b) {
    if (a.nodes.size() != b.nodes.size()) {
        return false;
    }

    for (const auto& [name, node] : a.nodes) {
        const auto found = b.nodes.find(name);
        if (found == b.nodes.end() || !SameNode(node, found->second)) {
            return false;
        }
    }

    return true;
}

bool Finite(const Rect& rect) {
    return std::isfinite(rect.x) && std::isfinite(rect.y) &&
           std::isfinite(rect.width) && std::isfinite(rect.height);
}

bool Finite(const Extent2D<F32>& extent) {
    return std::isfinite(extent.x) && std::isfinite(extent.y);
}

}  // namespace

struct AgentExecutor::Binding {
    std::mutex mutex;
    std::shared_ptr<Flowgraph> graph;
};

struct AgentExecutor::Shared final : Agent::Tools::Host {
    Agent::Tools tools{*this};

    struct Reply {
        Agent::Tools::Output output;
        std::optional<Agent::Tools::Layout> layout;
        Agent::Tools::Call call;
    };

    enum class Phase : U8 {
        Queued,
        Running,
        Expired,
    };

    struct Request {
        std::string name;
        std::string args;
        std::shared_ptr<std::promise<Reply>> result;
        Agent::Cancellation shouldAbort;
        std::shared_ptr<std::atomic<Phase>> phase;
        std::shared_ptr<Binding> binding;

        bool aborted() const {
            return shouldAbort && shouldAbort();
        }

        bool cancelled() const {
            return phase->load() == Phase::Expired || aborted();
        }

        bool start() const {
            Phase expected = Phase::Queued;
            return phase->compare_exchange_strong(expected, Phase::Running);
        }

        bool expire() const {
            Phase expected = Phase::Queued;
            return phase->compare_exchange_strong(expected, Phase::Expired);
        }

        void finish(Agent::Tools::Output output,
                    std::optional<Agent::Tools::Layout> layout = {},
                    Agent::Tools::Call call = {}) const {
            result->set_value({std::move(output), std::move(layout), std::move(call)});
        }
    };

    struct Completed {
        Request request;
        Agent::Tools::Call call;
        Agent::Tools::Output output;
        std::optional<Agent::Tools::Layout> layout;
        U64 frames = 0;
        U64 stableFrames = 0;
    };

    std::mutex mutex;
    std::deque<Request> queue;
    std::deque<Completed> completed;
    std::deque<Completed> awaitingLayout;
    std::deque<std::string> failedFocus;
    std::shared_ptr<Instance> instance;
    std::atomic<bool> stopped{false};

    std::unordered_map<std::string, std::shared_ptr<Flowgraph>> graphs;
    std::shared_ptr<Flowgraph> focused;
    std::set<std::string> reserved;

    bool has(const std::string& name) override {
        return graphs.contains(name) || reserved.contains(name);
    }

    Result create(const std::string& name, std::shared_ptr<Flowgraph>& created) override {
        std::shared_ptr<Instance> target;
        {
            std::lock_guard lock(mutex);
            target = instance;
        }

        if (!target) {
            return Result::ERROR;
        }
        return target->flowgraphCreate(name, {}, created);
    }

    std::optional<F32> width(Block::NodeSize size) override {
        return DefaultNodeWidth(size);
    }

    std::optional<std::vector<DocMatch>> searchDocs(const std::string& query, U64 limit) override {
        const auto& docs = Docs::Bundled();
        const auto& pages = docs.pages();

        std::vector<DocMatch> matches;
        if (query.find_first_not_of(" \t\n") == std::string::npos) {
            for (const auto& page : pages) {
                matches.push_back({page.slug, page.title, {}, {}, page.description});
            }
            return matches;
        }

        for (const auto& hit : docs.search(query, limit)) {
            const auto& page = pages.at(hit.page);
            matches.push_back({page.slug, page.title, hit.heading, hit.anchor, hit.snippet});
        }
        return matches;
    }

    std::optional<DocPage> readDocs(const std::string& slug) override {
        const auto& docs = Docs::Bundled();
        const auto index = docs.find(slug);
        if (!index) {
            return std::nullopt;
        }

        const auto& page = docs.pages().at(*index);
        return DocPage{page.slug, page.title, page.description, page.body};
    }

    void failFocus(const std::string& name) {
        std::lock_guard lock(mutex);
        failedFocus.push_back(name);
    }

    void complete(const Request& request, const Agent::Tools::Call& call, Agent::Tools::Output output) {
        if (!call.definition || !call.definition->flags.reportsLayout) {
            request.finish(std::move(output));
            return;
        }

        std::lock_guard lock(mutex);
        if (!stopped.load() && !request.cancelled()) {
            completed.push_back({request, call, std::move(output), std::nullopt, 0, 0});
        }
    }

    Result run(const Request& request, Agent::Tools::Call& call) {
        if (stopped.load() || !request.start()) {
            if (!call.create.empty()) {
                failFocus(call.create);
            }
            return Result::SUCCESS;
        }

        if (request.aborted()) {
            if (!call.create.empty()) {
                failFocus(call.create);
            }
            request.finish(Agent::Tools::Error("tool cancelled before it started, so nothing changed"));
            return Result::SUCCESS;
        }

        Agent::Tools::Output output;
        try {
            output = tools.run(call);
        } catch (...) {
            output = Agent::Tools::Error("Tool execution failed unexpectedly.", true);
        }

        const bool creating = !call.create.empty();
        if (request.binding && creating && call.flowgraph) {
            std::lock_guard lock(request.binding->mutex);
            request.binding->graph = call.flowgraph;
        }

        if (creating && !call.flowgraph) {
            failFocus(call.create);
            request.finish(std::move(output));
            return Result::ERROR;
        }

        complete(request, call, std::move(output));
        return Result::SUCCESS;
    }

    static void Submit(const std::shared_ptr<Shared>& shared,
                       DefaultCompositorState& state,
                       const DefaultCompositorCallbacks& callbacks,
                       const Request& request) {
        auto target = shared->focused;
        if (request.binding) {
            std::lock_guard lock(request.binding->mutex);
            target = request.binding->graph;

            const auto& graphs = shared->graphs;
            const bool closed = target && std::none_of(graphs.begin(), graphs.end(), [&](const auto& item) {
                return item.second == target;
            });
            if (closed) {
                request.finish(Agent::Tools::Error("The graph selected for this turn was closed."));
                return;
            }
        }

        Agent::Tools::Call call;
        try {
            call = shared->tools.prepare(target, request.name, request.args);
        } catch (...) {
            request.finish(Agent::Tools::Error(
                "Could not prepare the tool arguments. No tool operation was started."));
            return;
        }

        if (call.output.has_value()) {
            request.finish(std::move(*call.output));
            return;
        }

        if (!call.create.empty()) {
            if (!state.system.instance) {
                request.finish(Agent::Tools::Error("no instance available"));
                return;
            }
            shared->reserved.insert(call.create);
            state.interface.pendingFocusedFlowgraph = call.create;
        }

        callbacks.enqueueCommand([shared = shared, request, call]() mutable -> Result {
            return shared->run(request, call);
        }, true);
    }
};

AgentExecutor::AgentExecutor(DefaultCompositorState& state, const DefaultCompositorCallbacks& callbacks) :
    shared(std::make_shared<Shared>()),
    state(state),
    callbacks(callbacks) {}

AgentExecutor::~AgentExecutor() {
    shutdown();
}

void AgentExecutor::shutdown() {
    shared->stopped.store(true);

    std::lock_guard lock(shared->mutex);
    shared->queue.clear();
    shared->completed.clear();
    shared->awaitingLayout.clear();
}

Agent::Session::ToolExecutor AgentExecutor::bind() {
    auto binding = std::make_shared<Binding>();
    if (const auto active = state.activeFlowgraph()) {
        binding->graph = state.flowgraph.items.at(*active);
    }

    return [this, binding](const auto& name, const auto& args, const auto& cancelled) {
        return execute(name, args, cancelled, binding);
    };
}

Agent::Session::ToolExecutor AgentExecutor::focused() {
    return [this](const auto& name, const auto& args, const auto& cancelled) {
        return execute(name, args, cancelled, nullptr);
    };
}

Agent::Tools::Output AgentExecutor::execute(const std::string& name,
                                            const std::string& args,
                                            const Agent::Cancellation& shouldAbort,
                                            std::shared_ptr<Binding> binding) {
    auto result = std::make_shared<std::promise<Shared::Reply>>();
    auto future = result->get_future();

    const Shared::Request request{
        name,
        args,
        result,
        shouldAbort,
        std::make_shared<std::atomic<Shared::Phase>>(Shared::Phase::Queued),
        std::move(binding),
    };

    struct Expire {
        const Shared::Request& request;

        ~Expire() {
            request.expire();
        }
    } expire{request};

    {
        std::lock_guard<std::mutex> lock(shared->mutex);
        if (shared->stopped.load()) {
            return Agent::Tools::Error("agent is shutting down");
        }
        if (request.cancelled()) {
            return Agent::Tools::Error("tool cancelled");
        }
        shared->queue.push_back(request);
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (future.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) {
        if (shared->stopped.load()) {
            return Agent::Tools::Error("agent is shutting down");
        }
        if (request.aborted()) {
            if (request.expire()) {
                return Agent::Tools::Error("tool cancelled before it started, so nothing changed");
            }
            return Agent::Tools::Error("tool cancelled while running", true);
        }
        if (std::chrono::steady_clock::now() >= deadline && request.expire()) {
            return Agent::Tools::Error("tool did not start within 30 seconds, so nothing changed");
        }
    }

    auto reply = future.get();
    if (reply.layout.has_value()) {
        Agent::Tools::Annotate(reply.call, reply.output, *reply.layout);
    }
    return std::move(reply.output);
}

void AgentExecutor::layout(const std::string& flowgraph, Agent::Tools::Layout layout) {
    std::erase_if(layout.nodes, [](const auto& entry) {
        const auto& node = entry.second;
        return !Finite(node.bounds) || node.bounds.empty() || !Finite(node.size);
    });
    for (auto& [_, node] : layout.nodes) {
        if (node.content.has_value() && !Finite(*node.content)) {
            node.content.reset();
        }
    }

    const auto& viewport = layout.viewport;
    const bool visible = viewport.has_value() && Finite(viewport->visible) && viewport->visible.width > 0.0f &&
                         viewport->visible.height > 0.0f;
    if (!visible) {
        layout.viewport.reset();
    }

    layouts[flowgraph] = std::move(layout);
}

void AgentExecutor::poll() {
    release(std::move(layouts));
    layouts.clear();
    dispatch();
}

void AgentExecutor::release(Layouts layouts) {
    using MeasuredLayout = std::optional<Agent::Tools::Layout>;
    const auto measured = [&](const std::shared_ptr<Flowgraph>& flowgraph) -> MeasuredLayout {
        for (const auto& [id, item] : state.flowgraph.items) {
            if (item != flowgraph) {
                continue;
            }

            const auto found = layouts.find(id);
            if (found != layouts.end()) {
                return found->second;
            }
        }
        return std::nullopt;
    };

    std::deque<Shared::Completed> ready;
    {
        std::lock_guard<std::mutex> lock(shared->mutex);

        std::deque<Shared::Completed> waiting;
        for (auto& item : shared->awaitingLayout) {
            auto current = measured(item.call.flowgraph);
            const bool measuredBoth = current.has_value() && item.layout.has_value();
            const bool same = measuredBoth && SameNodes(*item.layout, *current);
            if (same) {
                item.stableFrames = item.stableFrames + 1;
            } else {
                item.stableFrames = 0;
            }
            item.layout = std::move(current);

            bool settled = !item.layout.has_value() || item.stableFrames >= kSettledFrames;
            if (!settled) {
                settled = ++item.frames >= kMaxSettleFrames || item.request.cancelled();
            }

            if (settled) {
                ready.push_back(std::move(item));
            } else {
                waiting.push_back(std::move(item));
            }
        }
        shared->awaitingLayout = std::move(waiting);

        for (auto& item : shared->completed) {
            shared->awaitingLayout.push_back(std::move(item));
        }
        shared->completed.clear();

        for (const auto& name : shared->failedFocus) {
            if (state.interface.pendingFocusedFlowgraph == name) {
                state.interface.pendingFocusedFlowgraph.reset();
            }
        }
        shared->failedFocus.clear();
    }

    for (auto& completed : ready) {
        if (!shared->stopped.load() && !completed.request.cancelled()) {
            completed.request.finish(std::move(completed.output),
                                     std::move(completed.layout),
                                     std::move(completed.call));
        }
    }
}

void AgentExecutor::dispatch() {
    std::deque<Shared::Request> pending;
    {
        std::lock_guard<std::mutex> lock(shared->mutex);
        pending.swap(shared->queue);
        shared->instance = state.system.instance;
    }

    if (pending.empty()) {
        return;
    }

    shared->graphs = state.flowgraph.items;
    for (const auto& focus : {state.interface.pendingFocusedFlowgraph, state.activeFlowgraph()}) {
        const auto it = focus ? state.flowgraph.items.find(*focus) : state.flowgraph.items.end();
        if (it != state.flowgraph.items.end()) {
            shared->focused = it->second;
            break;
        }
    }

    for (const auto& request : pending) {
        if (shared->stopped.load() || request.cancelled()) {
            continue;
        }
        Shared::Submit(shared, state, callbacks, request);
    }

    shared->graphs.clear();
    shared->focused.reset();
    shared->reserved.clear();
}

}  // namespace Jetstream
