#include "jetstream/home.hh"

#include "jetstream/logger.hh"

#include <algorithm>
#include <mutex>
#include <unordered_set>
#include <utility>

namespace Jetstream {

struct Home::Gallery::Impl {
    struct Slot {
        std::string origin;
        std::vector<Section> sections;
    };

    mutable std::mutex mutex;
    std::vector<Slot> slots;
    U64 revision = 0;

    bool sectionTaken(const std::string& origin, const std::string& key) const {
        for (const auto& slot : slots) {
            if (slot.origin == origin) {
                continue;
            }
            for (const auto& section : slot.sections) {
                if (section.key == key) {
                    return true;
                }
            }
        }
        return false;
    }

    bool flowgraphTaken(const std::string& origin, const std::string& key) const {
        for (const auto& slot : slots) {
            if (slot.origin == origin) {
                continue;
            }
            for (const auto& section : slot.sections) {
                for (const auto& flowgraph : section.flowgraphs) {
                    if (flowgraph.key == key) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
};

struct Home::Impl {
    Gallery gallery;
};

Result Home::Gallery::Flowgraph::Parse(const std::string& key, std::string content, Flowgraph& flowgraph) {
    flowgraph = {};
    Parser::Map data;
    JST_CHECK(Parser::YamlDecode(content, data));
    JST_CHECK(flowgraph.deserialize(data));
    flowgraph.key = key;
    flowgraph.content = std::move(content);
    return Result::SUCCESS;
}

Home::Gallery::Gallery() : impl(std::make_unique<Impl>()) {}

Home::Gallery::~Gallery() = default;

Result Home::Gallery::publish(const std::string& origin, std::vector<Section> sections) {
    if (origin.empty()) {
        JST_ERROR("[HOME] Gallery sections need an origin.");
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> guard(impl->mutex);

    std::unordered_set<std::string> sectionKeys;
    std::unordered_set<std::string> flowgraphKeys;
    for (auto& section : sections) {
        if (section.title.empty()) {
            section.title = section.key;
        }
        if (section.key.empty() || !sectionKeys.insert(section.key).second ||
            impl->sectionTaken(origin, section.key)) {
            JST_ERROR("[HOME] Gallery section '{}' from '{}' is empty or already taken.", section.key, origin);
            return Result::ERROR;
        }
        for (auto& flowgraph : section.flowgraphs) {
            flowgraph.featured = flowgraph.featured && origin == Origin;
            if (flowgraph.key.empty() || !flowgraphKeys.insert(flowgraph.key).second ||
                impl->flowgraphTaken(origin, flowgraph.key)) {
                JST_ERROR("[HOME] Gallery flowgraph '{}' from '{}' is empty or already taken.", flowgraph.key, origin);
                return Result::ERROR;
            }
        }
    }

    auto& slots = impl->slots;
    auto slot = std::find_if(slots.begin(), slots.end(), [&](const auto& item) {
        return item.origin == origin;
    });
    if (sections.empty()) {
        if (slot != slots.end()) {
            slots.erase(slot);
            ++impl->revision;
        }
        return Result::SUCCESS;
    }
    if (slot != slots.end()) {
        slot->sections = std::move(sections);
    } else if (origin == Origin) {
        slots.insert(slots.begin(), {origin, std::move(sections)});
    } else {
        slots.push_back({origin, std::move(sections)});
    }
    ++impl->revision;
    return Result::SUCCESS;
}

std::vector<Home::Gallery::Section> Home::Gallery::sections() const {
    std::lock_guard<std::mutex> guard(impl->mutex);
    std::vector<Section> result;
    for (const auto& slot : impl->slots) {
        result.insert(result.end(), slot.sections.begin(), slot.sections.end());
    }
    return result;
}

std::optional<Home::Gallery::Flowgraph> Home::Gallery::flowgraph(const std::string& key) const {
    std::lock_guard<std::mutex> guard(impl->mutex);
    for (const auto& slot : impl->slots) {
        for (const auto& section : slot.sections) {
            for (const auto& flowgraph : section.flowgraphs) {
                if (flowgraph.key == key) {
                    return flowgraph;
                }
            }
        }
    }
    return std::nullopt;
}

U64 Home::Gallery::revision() const {
    std::lock_guard<std::mutex> guard(impl->mutex);
    return impl->revision;
}

Home::Home() : impl(std::make_unique<Impl>()) {}

Home::~Home() = default;

Home& Home::Get() {
    static Home home;
    return home;
}

Home::Gallery& Home::gallery() {
    return impl->gallery;
}

}  // namespace Jetstream
