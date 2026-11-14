#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DEPENDENCIES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DEPENDENCIES_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/dependencies.hh"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace Jetstream {

struct DependencyReviewModalPresenter {
    const PresenterContext& context;

    explicit DependencyReviewModalPresenter(const PresenterContext& context);
    DependencyReviewView::Config build() const;

 private:
    static constexpr std::string_view moduleSuffix = "-python";

    std::string flowgraphIdFor(const std::weak_ptr<Flowgraph::View>& weakView) const;
    std::string requestedBy(const PythonDependencyRequestEntry& dependency) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DEPENDENCIES_HH
