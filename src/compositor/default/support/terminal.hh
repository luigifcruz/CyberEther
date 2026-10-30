#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_TERMINAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_TERMINAL_HH

#include "jetstream/render/sakura/components/terminal.hh"
#include "jetstream/surface.hh"
#include "jetstream/types.hh"

#include <memory>
#include <string>
#include <vector>

namespace Jetstream {

class TerminalSession {
 public:
    struct Theme {
        ColorRGBA<F32> foreground = {0.9f, 0.9f, 0.9f, 1.0f};
        ColorRGBA<F32> background = {0.0f, 0.0f, 0.0f, 1.0f};

        bool operator==(const Theme&) const = default;
    };

    struct Config {
        std::string program;
        std::vector<std::string> arguments;
        std::string directory;
        Theme theme;
    };

    TerminalSession();
    ~TerminalSession();

    TerminalSession(const TerminalSession&) = delete;
    TerminalSession& operator=(const TerminalSession&) = delete;

    static bool Supported();
    static bool AltSendsEscape();

    Result start(const Config& config);
    void stop();
    void poll();

    bool running() const;
    const std::string& title() const;
    const Sakura::Terminal::Screen& screen() const;
    U64 revision() const;

    void theme(const Theme& theme);
    void resize(const Sakura::Terminal::Grid& grid);
    void key(const KeyEvent& event);
    void text(const std::string& text);
    void paste(const std::string& text);
    void scroll(I32 rows, Extent2D<F32> position, KeyModifiers modifiers = {});
    void scrollTo(U64 row);
    void select(const Sakura::Terminal::Selection& selection);
    void pointer(const MouseEvent& event);
    void focus(bool focused);
    void search(const std::string& needle);
    void searchStep(bool older);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_TERMINAL_HH
