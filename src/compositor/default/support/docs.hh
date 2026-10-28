#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_DOCS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_DOCS_HH

#include "jetstream/types.hh"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Jetstream {

namespace Resources {

const std::vector<std::pair<std::string_view, std::string_view>>& BundledDocs();

}  // namespace Resources

class JETSTREAM_API Docs {
 public:
    static constexpr const char* Home = "overview";
    static constexpr const char* LinkPrefix = "/docs/";
    static constexpr const char* WebBase = "https://cyberether.org/docs/";

    struct Page {
        std::string slug;
        std::string title;
        std::string description;
        std::string category;
        std::string parent;
        I64 order = 100;
        U64 depth = 0;
        std::string body;
    };

    struct Group {
        std::string title;
        std::vector<U64> pages;
    };

    struct Link {
        std::string slug;
        std::string anchor;
    };

    static const Docs& Bundled();
    static Docs Parse(const std::vector<std::pair<std::string_view, std::string_view>>& files);
    static std::optional<Link> Resolve(const std::string& url);
    static std::string WebUrl(const Link& link);
    static std::string Anchor(const std::string& heading);

    const std::vector<Page>& pages() const;
    const std::vector<Group>& groups() const;
    std::optional<U64> find(const std::string& slug) const;

 private:
    std::vector<Page> ordered;
    std::vector<Group> grouped;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_DOCS_HH
