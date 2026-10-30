#ifndef JETSTREAM_ADAPTER_GENERIC_HH
#define JETSTREAM_ADAPTER_GENERIC_HH

#include "jetstream/logger.hh"
#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/input.hh"
#include "jetstream/render/base/window_metrics.hh"
#include "jetstream/surface.hh"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Viewport {

struct Config {
    /// @brief Whether vertical synchronization (vsync) is enabled. Only applicable to headed viewports.
    bool vsync = true;

    /// @brief The title of the application or window.
#if defined(JST_OS_BROWSER)
    std::string title = "CyberEther Web";
#else
    std::string title = "CyberEther";
#endif

    /// @brief The size of the application or window.
    Extent2D<U64> size = {1920, 1080};

    /// @brief The framerate of the remote viewport.
    U64 framerate = 60;

    bool integratedTitlebar = false;
};

class JETSTREAM_API Generic {
 public:
    class Platform {
     public:
        virtual ~Platform() = default;

        virtual F64 time() const = 0;
        virtual void poll() = 0;
        virtual void setCursor(SurfaceCursor cursor) = 0;
        virtual std::string clipboardText() const = 0;
        virtual void setClipboardText(const std::string& text) = 0;
    };

    explicit Generic(const Config& config);
    virtual ~Generic();

    virtual std::string id() const = 0;
    virtual std::string name() const = 0;
    virtual DeviceType device() const = 0;

    virtual Result create() = 0;
    virtual Result destroy() = 0;

    virtual Extent2D<F32> displaySize() const;
    virtual Extent2D<F32> framebufferSize() const;
    virtual Extent2D<F32> contentScale() const;
    virtual Render::WindowMetrics::Policy scalePolicy() const;

    Render::WindowMetrics metrics(F32 interfaceScale) const;

    virtual Result waitEvents() = 0;
    virtual Result pollEvents() = 0;
    virtual bool keepRunning() = 0;

    virtual Render::TitlebarChrome titlebarChrome() const;
    virtual void setIntegratedTitlebar(bool enabled);
    bool integratedTitlebar() const;
    virtual void beginWindowDrag();
    virtual void zoomWindow();

    virtual bool macOSBehaviors() const;
    virtual bool retainedPresent() const;
    virtual void waitIdle(F64 seconds);

    F64 inputTime() const;
    void setCursor(SurfaceCursor cursor);
    std::string clipboardText() const;
    void setClipboardText(const std::string& text);
    void requestRedraw();
    void refreshSnapshot();
    bool consumeRedraw();

    Render::InputQueue& input();

    Result addFileDropEvent(std::vector<std::string> paths);

    std::vector<std::string> takeDroppedFiles();

 protected:
    Config config;

    void attachPlatform(std::unique_ptr<Platform> next);
    void detachPlatform();
    void pumpPlatform();

 private:
    struct Snapshot {
        Extent2D<F32> displaySize;
        Extent2D<F32> framebufferSize;
        Extent2D<F32> contentScale;
    };

    std::unique_ptr<Platform> platform;
    mutable std::mutex snapshotMutex;
    std::optional<Snapshot> snapshot;
    std::string clipboard;
    Render::InputQueue inputQueue;
    std::atomic<bool> redrawRequested = true;
    std::mutex droppedFilesMutex;
    std::vector<std::string> droppedFiles;
};

template<DeviceType DeviceId>
class Adapter : public Generic {};

}  // namespace Jetstream::Viewport

#endif
