#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodePin : public Component {
    enum class Direction {
        Input,
        Output,
    };

    struct Config {
        std::string id;
        Direction direction = Direction::Input;
        std::string label;
        std::string help;
        bool enableDetach = false;
        Shape dataShape;
        Shape dataStride;
        DataType dataType = DataType::None;
        DeviceType dataDevice = DeviceType::None;
        U64 dataOffsetBytes = 0;
        bool dataContiguous = false;
        std::vector<std::vector<std::string>> dataAttributes;
    };

    NodePin();
    ~NodePin();

    NodePin(NodePin&&) noexcept;
    NodePin& operator=(NodePin&&) noexcept;

    NodePin(const NodePin&) = delete;
    NodePin& operator=(const NodePin&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
