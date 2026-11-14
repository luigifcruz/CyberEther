#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DOCS_SEARCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DOCS_SEARCH_HH

#include "../../support/docs.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "tools/text.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct DocsSearchView {
    static constexpr U64 ResultLimit = 6;
    static constexpr F32 Width = 720.0f;
    static constexpr F32 FieldHeight = 44.0f;
    static constexpr F32 CardPadding = 10.0f;
    static constexpr F32 CardHeight = 460.0f;
    static constexpr F32 RowPadding = 8.0f;
    static constexpr F32 RowGap = 2.0f;
    static constexpr F32 SnippetSize = 14.0f;

    struct Config {
        const Docs* docs = nullptr;
        std::string query;
        Sakura::Callback<void(const std::string&)> onQueryChange;
        Sakura::Callback<void(const std::string&, const std::string&)> onOpen;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    Sakura::Component& component();
    Sakura::ModalLayer::Config layer();

 private:
    struct Result {
        Sakura::Text page;
        Sakura::Text chevron;
        Sakura::Text heading;
        Sakura::Stack title;
        Sakura::HighlightText snippet;
        Sakura::Stack content;
        Sakura::Div card;
    };

    void updateField();
    void updateResults();
    void updateResult(U64 index);
    void step(int delta);
    void open(U64 index);
    void close();

    Config config;
    const Docs* searchedDocs = nullptr;
    std::string searchedQuery;
    std::vector<Docs::Hit> hits;
    U64 selected = 0;

    Sakura::PillField field;
    Sakura::PillIconButton closeButton;
    Sakura::Stack fieldRow;
    std::vector<std::unique_ptr<Result>> results;
    Sakura::Text empty;
    Sakura::Spacer emptyTop;
    Sakura::Spacer emptyBottom;
    Sakura::Stack emptyState;
    Sakura::Stack list;
    Sakura::ScrollArea scroll;
    Sakura::Div card;
    Sakura::Stack column;
    Sakura::Shortcuts keys;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DOCS_SEARCH_HH
