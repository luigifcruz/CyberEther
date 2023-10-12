#ifndef JETSTREAM_INTERFACE_HH
#define JETSTREAM_INTERFACE_HH

#include <unordered_map>
#include <unordered_set>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"

namespace Jetstream {

class JETSTREAM_API Interface {
 public:
    virtual ~Interface() = default;

    struct Config {
        F32 nodeWidth = 0.0f;
        bool viewEnabled = false;
        bool previewEnabled = false;
        bool controlEnabled = false;
        Size2D<F32> nodePos = {0.0f, 0.0f};

        JST_SERDES(
            JST_SERDES_VAL("nodeWidth", nodeWidth);
            JST_SERDES_VAL("viewEnabled", viewEnabled);
            JST_SERDES_VAL("previewEnabled", previewEnabled);
            JST_SERDES_VAL("controlEnabled", controlEnabled);
            JST_SERDES_VAL("nodePos", nodePos);
        );
    };

    constexpr const Config& getConfig() const {
        return _config;
    }

    constexpr const Config& getInterfaceConfig() const {
        return _config;
    }

    // Metadata Methods

    constexpr const bool& complete() const {
        return _complete;
    }

    constexpr const std::string& error() const {
        return _error;
    }

    constexpr const Locale& locale() const {
        return _locale;
    }

    virtual constexpr Device device() const = 0;
    virtual std::string_view name() const = 0;
    virtual std::string_view prettyName() const = 0;

    std::string title() const {
        return fmt::format("{} ({})", prettyName(), locale());
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

 protected:
    constexpr Instance& instance() {
        return *_instance;
    }

    void setInterfaceConfig(const Config& config) {
        _config = config;
    }

    void setLocale(const Locale& locale) {
        _locale = locale;
    }

    void setComplete(const bool& complete) {
        _complete = complete;
    }

    void pushError(const std::string& error) {
        _error += ((!_error.empty()) ? "\n" : "") + error;
    }

    void setInstance(Instance* instance) {
        _instance = instance;
    }

    constexpr Config& getConfig() {
        return _config;
    }

    constexpr Config& getInterfaceConfig() {
        return _config;
    }

    friend Instance;
    friend class Compositor;

 private:
    Config _config;
    Locale _locale;
    
    bool _complete;
    std::string _error;

    Instance* _instance;
};

}  // namespace Jetstream

#endif
