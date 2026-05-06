#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BASE_HH

#include "benchmark.hh"
#include "flowgraph.hh"
#include "remote.hh"
#include "settings.hh"
#include "workbench.hh"

#include "jetstream/logger.hh"

#include <memory>
#include <tuple>
#include <type_traits>
#include <variant>

namespace Jetstream {

class DefaultActions {
 public:
    DefaultActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks) {
        workbench = std::make_shared<WorkbenchActions>(state, callbacks);
        settings = std::make_shared<SettingsActions>(state, callbacks);
        benchmark = std::make_shared<BenchmarkActions>(state, callbacks);
        remote = std::make_shared<RemoteActions>(state, callbacks);
        flowgraph = std::make_shared<FlowgraphActions>(state, callbacks);
    }

    Result handle(const Mail& mail) {
        Result result = Result::SUCCESS;
        if (Handle(*workbench, mail, result)) {
            return result;
        }
        if (Handle(*settings, mail, result)) {
            return result;
        }
        if (Handle(*benchmark, mail, result)) {
            return result;
        }
        if (Handle(*remote, mail, result)) {
            return result;
        }
        if (Handle(*flowgraph, mail, result)) {
            return result;
        }

        JST_ERROR("[COMPOSITOR_IMPL_DEFAULT] Unhandled mail.");
        return Result::ERROR;
    }

 private:
    template<class Action, class... Signals>
    static bool Handle(Action& action, const Mail& mail, Result& result, std::tuple<Signals...>*) {
        return std::visit([&](const auto& msg) {
            using Signal = std::decay_t<decltype(msg)>;
            if constexpr ((std::is_same_v<Signal, Signals> || ...)) {
                result = action.handle(msg);
                return true;
            }
            return false;
        }, mail);
    }

    template<class Action>
    static bool Handle(Action& action, const Mail& mail, Result& result) {
        return Handle(action, mail, result, static_cast<typename Action::Filter*>(nullptr));
    }

    std::shared_ptr<BenchmarkActions> benchmark;
    std::shared_ptr<FlowgraphActions> flowgraph;
    std::shared_ptr<RemoteActions> remote;
    std::shared_ptr<SettingsActions> settings;
    std::shared_ptr<WorkbenchActions> workbench;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BASE_HH
