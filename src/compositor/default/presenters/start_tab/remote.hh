#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_REMOTE_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/start_tab/remote.hh"

#include <algorithm>
#include <cctype>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace Jetstream {

struct RemoteSectionPresenter {
    const PresenterContext& context;

    explicit RemoteSectionPresenter(const PresenterContext& context);
    std::optional<RemoteSectionView::Config> build() const;

 private:
    static std::string Code(const std::string& sessionId);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_REMOTE_HH
