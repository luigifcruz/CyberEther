#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH

#include "jetstream/render/sakura/base.hh"

#include "editor/config/base.hh"

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedConfig {
    struct Config {
        std::string id;
        std::string title;
        std::vector<FlowgraphConfigFieldConfig> configFields;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);
        window.update({
            .id = this->config.id,
            .title = this->config.title,
            .size = {320.0f, 360.0f},
            .padding = Extent2D<F32>{8.0f, 8.0f},
            .onClose = this->config.onClose,
        });

        fields.resize(this->config.configFields.size());
        for (U64 i = 0; i < fields.size(); ++i) {
            fields[i].update(this->config.configFields[i]);
        }
        fieldGrid.update({
            .id = this->config.id + ":fields",
        });

        contentChildren.clear();
        std::vector<std::optional<U64>> fieldLayoutItems(fields.size());
        Sakura::VStack::Config contentLayoutConfig{
            .id = this->config.id + ":content",
            .fill = true,
        };
        auto addContent = [this, &contentLayoutConfig](std::string id,
                                                       std::optional<Sakura::VStack::Flex> flex,
                                                       Sakura::VStack::Child child) {
            const U64 index = contentLayoutConfig.items.size();
            contentLayoutConfig.items.push_back({
                .id = std::move(id),
                .flex = flex,
                .gap = Sakura::NodeField::Gap,
            });
            contentChildren.push_back(std::move(child));
            return index;
        };

        for (U64 i = 0; i < fields.size();) {
            if (fields[i].isSimple()) {
                std::string id = "fields";
                std::vector<Sakura::NodeFieldGrid::Item> items;
                do {
                    const auto& simpleConfig = this->config.configFields[i];
                    id += ":" + simpleConfig.id + ":" + std::to_string(Parser::Hash(simpleConfig.format));
                    items.push_back({
                        .child = [this, i](const Sakura::Context& ctx) {
                            fields[i].render(ctx);
                        },
                    });
                    ++i;
                } while (i < fields.size() && fields[i].isSimple());
                addContent(std::move(id), std::nullopt,
                           [this, items = std::move(items)](const Sakura::Context& ctx) {
                               fieldGrid.render(ctx, items);
                           });
                continue;
            }

            const auto& fieldConfig = this->config.configFields[i];
            const auto spec = fields[i].heightSpec();
            std::optional<Sakura::VStack::Flex> flex;
            if (spec.policy == FlowgraphNodeHeightPolicy::FillRemaining) {
                flex = Sakura::VStack::Flex{
                    .minimum = spec.minimum,
                    .grow = spec.grow,
                };
            }
            fieldLayoutItems[i] = addContent("field:" + fieldConfig.id + ":" + std::to_string(Parser::Hash(fieldConfig.format)),
                                             flex,
                                             [this, i](const Sakura::Context& ctx) {
                                                 fields[i].render(ctx);
                                             });
            ++i;
        }

        contentLayout.update(std::move(contentLayoutConfig));
        const auto* contentLayoutState = contentLayout.layout();
        for (U64 i = 0; i < fields.size(); ++i) {
            std::optional<F32> allocated;
            if (contentLayoutState && fieldLayoutItems[i].has_value()) {
                allocated = contentLayoutState->itemHeight(*fieldLayoutItems[i]);
                const auto spec = fields[i].heightSpec();
                if (allocated.has_value() && spec.policy == FlowgraphNodeHeightPolicy::FillRemaining) {
                    allocated = std::max(*allocated, spec.minimum);
                }
            }
            fields[i].setAllocatedHeight(allocated);
        }
    }

    void render(const Sakura::Context& ctx) {
        window.render(ctx, [this](const Sakura::Context& ctx) {
            contentLayout.render(ctx, contentChildren);
        });
    }

 private:
    Config config;
    Sakura::Window window;
    Sakura::NodeFieldGrid fieldGrid;
    Sakura::VStack contentLayout;
    Sakura::VStack::Children contentChildren;
    std::vector<FlowgraphConfigFieldInstance> fields;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH
