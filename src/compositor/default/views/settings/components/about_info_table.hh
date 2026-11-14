#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH

#include "../../components/section_group.hh"

#include "jetstream/render/sakura/base.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct AboutInfoTable : public SectionGroup {
    struct Config {
        std::string title;
        std::vector<std::vector<std::string>> rows;
    };

    void update(Config config);

 private:
    std::vector<std::unique_ptr<Sakura::Text>> values;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH
