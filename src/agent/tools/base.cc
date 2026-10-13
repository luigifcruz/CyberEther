#include "support/base.hh"

#include "tools/naming.hh"

namespace Jetstream::Agent {

namespace Detail {

namespace {

Tools::Output Dispatch(Host& host, const Call& call) {
    try {
        const auto args = Json::parse(call.args, nullptr, false);
        if (!args.is_object()) {
            return Response::Failure("Arguments must be a JSON object.");
        }

        const auto* entry = Toolbox::Find(call.tool);
        if (!entry) {
            return Response::Failure(jst::fmt::format("Unknown tool '{}'.", call.tool));
        }

        return entry->execute({host, call.flowgraph, call.create, args});
    } catch (const std::exception& error) {
        return Response::Failure(jst::fmt::format("tool '{}' failed: {}", call.tool, error.what()));
    } catch (...) {
        return Response::Failure(jst::fmt::format("tool '{}' failed with an unknown error", call.tool));
    }
}

Tools::Output CreateFlowgraphError(const std::string& name) {
    return Response::Reply(Json({
        {"error", jst::fmt::format("Could not create flowgraph '{}'.", name)},
        {"code", "CREATE_FLOWGRAPH_FAILED"},
    }));
}

}  // namespace

Entry Entry::FromJson(Definition definition, Json (*execute)(const Graph&, const Json&)) {
    return {std::move(definition), [execute](const Invocation& call) {
        return Response::Reply(execute(call.flowgraph, call.args));
    }};
}

Entry Entry::FromJson(Definition definition, Json (*execute)(const Invocation&)) {
    return {std::move(definition), [execute](const Invocation& call) {
        return Response::Reply(execute(call));
    }};
}

const std::vector<Entry>& Toolbox::All() {
    static const auto entries = [] {
        const auto factories = {
            Catalog,
            Inspect,
            Patch,
            Text,
            Connections,
            Environment,
            Capture,
        };

        std::vector<Entry> result;
        for (auto factory : factories) {
            for (auto& entry : factory()) {
                result.push_back(std::move(entry));
            }
        }
        return result;
    }();
    return entries;
}

const Entry* Toolbox::Find(const std::string& name) {
    for (const auto& entry : All()) {
        if (entry.definition.function.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace Detail

const std::vector<Tools::Definition>& Tools::Definitions() {
    static const auto definitions = [] {
        std::vector<Definition> result;
        for (const auto& entry : Detail::Toolbox::All()) {
            result.push_back(entry.definition);
        }
        return result;
    }();
    return definitions;
}

std::vector<Provider::Function> Tools::Functions(bool captureSupported) {
    std::vector<Provider::Function> tools;
    for (const auto& definition : Definitions()) {
        if (definition.flags.needsCapture && !captureSupported) {
            continue;
        }
        tools.push_back(definition.function);
    }
    return tools;
}

Tools::Output Tools::Error(const std::string& message, bool mayHaveApplied) {
    auto body = Detail::Response::Error(message);
    if (mayHaveApplied) {
        body["may_have_applied"] = true;
    }
    return Detail::Response::Reply(body);
}

void Tools::Annotate(const Call& call, Tools::Output& output, const Layout& layout) {
    if (!call.definition || !call.definition->flags.reportsLayout) {
        return;
    }

    auto body = Detail::Json::parse(output.content, nullptr, false);
    if (!body.is_object()) {
        return;
    }

    Detail::Geometry::Annotate(body, layout);

    const auto* entry = Detail::Toolbox::Find(call.tool);
    if (entry && entry->annotate) {
        entry->annotate(body, Detail::Json::parse(call.args, nullptr, false), layout);
    }

    output.content = body.dump();
}

Tools::Tools(Host& host) : host(host) {}

Tools::Call Tools::prepare(const std::shared_ptr<Flowgraph>& target, const std::string& name,
                           const std::string& args) const {
    Call call{.tool = name, .args = args};

    const auto* entry = Detail::Toolbox::Find(name);
    if (!entry) {
        return call;
    }

    call.definition = &entry->definition;
    if (entry->definition.flags.needsGraph) {
        call.flowgraph = target;
    }

    if (call.flowgraph || !entry->newGraph) {
        return call;
    }

    const auto graph = entry->newGraph(Detail::Json::parse(args, nullptr, false));
    if (!graph.error.empty()) {
        call.output = Detail::Response::Failure(graph.error);
    } else if (!graph.name.empty()) {
        call.create = Naming::UniqueName(graph.name, [this](const std::string& candidate) {
            return host.has(candidate);
        });
    }

    return call;
}

Tools::Output Tools::run(Call& call, const Cancellation& cancel) const {
    if (call.output.has_value()) {
        return *call.output;
    }

    if (cancel && cancel()) {
        return Detail::Response::Failure("tool cancelled");
    }

    if (!call.create.empty()) {
        std::shared_ptr<Flowgraph> created;
        if (host.create(call.create, created) != Result::SUCCESS) {
            return Detail::CreateFlowgraphError(call.create);
        }
        call.flowgraph = std::move(created);
    }

    return Detail::Dispatch(host, call);
}

Tools::Output Tools::execute(const std::shared_ptr<Flowgraph>& target, const std::string& name,
                             const std::string& args, const Cancellation& cancel) const {
    auto call = prepare(target, name, args);
    return run(call, cancel);
}

}  // namespace Jetstream::Agent
