#ifndef JETSTREAM_BLOCK_HH
#define JETSTREAM_BLOCK_HH

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/base.hh"


namespace Jetstream {

class JETSTREAM_API Block {
 public:
    virtual ~Block() = default;

    // Configuration.

    struct State {
        F32 nodeWidth = 0.0f;
        bool viewEnabled = false;
        bool previewEnabled = false;
        bool controlEnabled = false;
        Size2D<F32> nodePos = {0.0f, 0.0f};

        JST_SERDES(nodeWidth, viewEnabled, previewEnabled, controlEnabled, nodePos);
    };

    constexpr const State& getState() const {
        return state;
    }

    // Fingerprint, manifest, and metadata.

    struct Fingerprint {
        std::string id;
        std::string device;
        std::string inputDataType;
        std::string outputDataType;

        struct Hash {
            U64 operator()(const Fingerprint& m) const {
                U64 h1 = std::hash<std::string>()(m.id);
                U64 h2 = std::hash<std::string>()(m.device);
                U64 h3 = std::hash<std::string>()(m.inputDataType);
                U64 h4 = std::hash<std::string>()(m.outputDataType);
                return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
            }
        };

        struct Equal {
            bool operator()(const Fingerprint& m1, const Fingerprint& m2) const {
                return m1.id == m2.id
                    && m1.device == m2.device
                    && m1.inputDataType == m2.inputDataType
                    && m1.outputDataType == m2.outputDataType;
            }
        };

        friend std::ostream& operator<<(std::ostream& os, const Fingerprint& m) {
            os << jst::fmt::format("{{device: '{}', id: '{}', inputDataType: '{}', outputDataType: '{}'}}", 
                                   m.device, m.id, m.inputDataType, m.outputDataType);
            return os;
        }
    };

    typedef std::function<Result(Instance&,
                                 const std::string& id,
                                 Parser::RecordMap& inputMap,
                                 Parser::RecordMap& configMap,
                                 Parser::RecordMap& stateMap)> Constructor;

    typedef std::unordered_map<Fingerprint,
                               Constructor,
                               Fingerprint::Hash,
                               Fingerprint::Equal> ConstructorManifest;

    struct Metadata {
        std::string title;
        std::string summary;
        std::string description;
        std::map<Device, std::vector<std::tuple<std::string, std::string>>> options;
    };

    typedef std::map<std::string, Metadata> MetadataManifest;

    // Construction methods.

    virtual Result create() = 0;
    virtual Result destroy() = 0;

    // Metadata methods.

    virtual std::string id() const = 0;
    virtual std::string name() const = 0;
    virtual constexpr Device device() const = 0;

    virtual std::string summary() const {
        return "Summary not available.";
    }

    virtual std::string description() const {
        return "Description not available.";
    }

    virtual std::string warning() const {
        return "";
    }

    constexpr const bool& complete() const {
        return _complete;
    }

    constexpr const std::string& error() const {
        return _error;
    }

    constexpr const Locale& locale() const {
        return _locale;
    }

    // Graphical Methods

    virtual void drawPreview(const F32&) {}
    virtual constexpr bool shouldDrawPreview() const {
        return false;
    }

    virtual void drawView() {}
    virtual constexpr bool shouldDrawView() const {
        return false;
    }

    virtual void drawControl() {}
    virtual constexpr bool shouldDrawControl() const {
        return false;
    }

    virtual void drawInfo() {}
    virtual constexpr bool shouldDrawInfo() const {
        return false;
    }

    // Helpers

    static std::pair<U64, U64> GetContentRegion() {
        ImVec2 contentRegion = ImGui::GetContentRegionAvail();
        return {
            static_cast<U64>(contentRegion.x),
            static_cast<U64>(contentRegion.y)
        };
    }

    static std::pair<F32, F32> GetRelativeMousePos(const std::pair<U64, U64> dim, const F32& zoom = 1.0f) {
        const auto& [x, y] = dim;

        ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
        ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();

        return {
            (((mousePositionAbsolute.x - screenPositionAbsolute.x) / x) * 2.0f - 1.0f) / zoom,
            (((mousePositionAbsolute.y - screenPositionAbsolute.y) / y) * 2.0f - 1.0f) / zoom,
        };
    }

    static std::pair<F32, F32> GetRelativeMouseTranslation(const std::pair<U64, U64> dim, const F32& zoom = 1.0f) {
        const auto& [dx, dy] = ImGui::GetMouseDragDelta(0);
        const auto& [x, y] = dim;

        return {
            ((dx * (1.0f / x)) * 2.0f) / zoom,
            ((dy * (1.0f / y)) * 2.0f) / zoom
        };
    }

 protected:
    constexpr Instance& instance() {
        return *_instance;
    }

    void setComplete(const bool& complete) {
        _complete = complete;
    }

    void setInstance(Instance* instance) {
        _instance = instance;
    }

    void setLocale(const Locale& locale) {
        _locale = locale;
    }

    void pushError(const std::string& error) {
        _error += ((!_error.empty()) ? "\n" : "") + error;
    }

    template<Device DeviceId, typename Type>
    static Result LinkOutput(const std::string& name,
                             Tensor<DeviceId, Type>& dst,
                             const Tensor<DeviceId, Type>& src) {
        dst.set_locale({
            src.locale().blockId, 
            src.locale().moduleId, 
            name
        });

        if (!dst.empty()) {
            JST_ERROR("The destination buffer should be empty during initialization.");
            return Result::ERROR;
        }

        dst = src;

        return Result::SUCCESS;
    }

    friend class Compositor;
    friend class Instance;

 private:
    State state;

    Locale _locale;
    bool _complete;
    std::string _error;

    Instance* _instance;
};

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::Block::Fingerprint> : ostream_formatter {};

#endif
