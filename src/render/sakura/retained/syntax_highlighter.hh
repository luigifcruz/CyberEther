#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_SYNTAX_HIGHLIGHTER_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_SYNTAX_HIGHLIGHTER_HH

#include <jetstream/types.hh>

#include <tree_sitter/api.h>

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct SyntaxHighlighter {
    using StyleId = U8;

    enum class Language : U8 {
        Python,
        Markdown,
        Yaml,
        Bash,
        Cpp,
        Json,
    };

    static constexpr U64 LanguageCount = 6;

    enum Style : StyleId {
        Default = 0,
        Comment = 1,
        Keyword = 2,
        String = 3,
        Number = 4,
        Function = 5,
        Type = 6,
        Constant = 7,
        Operator = 8,
        Property = 9,
    };

    static constexpr StyleId StyleCount = 9;

    static std::optional<Language> LanguageForTag(std::string_view tag);
    static bool IsCommentOrString(StyleId id);

    SyntaxHighlighter();
    ~SyntaxHighlighter();

    SyntaxHighlighter(const SyntaxHighlighter&) = delete;
    SyntaxHighlighter& operator=(const SyntaxHighlighter&) = delete;

    const std::vector<std::vector<StyleId>>& styles(const std::vector<std::string>& lines, U64 revision,
                                                    Language language);
    std::vector<std::vector<StyleId>> highlight(const std::vector<std::string>& lines, Language language);
    bool rootNode(const std::vector<std::string>& lines, U64 revision, Language language, TSNode& root);

 private:
    TSParser* parser = nullptr;
    std::array<TSQuery*, LanguageCount> queries = {};
    std::array<bool, LanguageCount> queryFailed = {};
    TSTree* tree = nullptr;
    std::vector<std::vector<StyleId>> cachedStyles;
    U64 cachedRevision = 0;
    Language cachedLanguage = Language::Python;
    Language activeLanguage = Language::Python;
    bool cacheValid = false;
    bool activeLanguageValid = false;

    bool ensureTreeSitter(Language language);
    void applyQuery(const TSQuery* query, std::vector<std::vector<StyleId>>& styles,
                    const std::vector<std::string>& lines, TSNode root) const;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_SYNTAX_HIGHLIGHTER_HH
