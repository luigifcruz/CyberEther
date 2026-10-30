#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/layout.hh"
#include "compositor/default/actions/terminal.hh"
#include "compositor/default/actions/workbench.hh"
#include "compositor/default/support/terminal.hh"
#include "environment.hh"
#include "tools/random.hh"

#include "jetstream/flowgraph.hh"
#include "jetstream/platform.hh"

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace Jetstream;

namespace {

constexpr const char* kTags = "[core][compositor][terminal]";

std::string RowText(const Sakura::Terminal::Row& row) {
    std::string text;
    U16 column = 0;
    for (const auto& span : row.spans) {
        if (span.column > column) {
            text.append(span.column - column, ' ');
        }
        text += span.text;
        column = span.column + span.width;
    }
    return text;
}

std::string ScreenText(const TerminalSession& session) {
    std::string text;
    for (const auto& row : session.screen().lines) {
        text += RowText(row);
        text += '\n';
    }
    return text;
}

bool PollUntil(TerminalSession& session, const std::function<bool()>& done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        session.poll();
        if (done()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

bool Shows(TerminalSession& session, const std::string& text) {
    return PollUntil(session, [&] { return ScreenText(session).find(text) != std::string::npos; });
}

TerminalSession::Config Shell() {
    return {.program = "/bin/sh", .arguments = {}, .directory = "/"};
}

bool StartShell(TerminalSession& session) {
    if (session.start(Shell()) != Result::SUCCESS) {
        return false;
    }
    session.text("printf 'shell%s\\n' ready\r");
    return Shows(session, "shellready");
}

std::string TileShape(const TileMeta& tiles) {
    if (IsTileLeaf(tiles)) {
        return tiles.pane.value_or("");
    }
    std::string shape = tiles.split.value_or("row") + "(";
    for (U64 index = 0; index < tiles.children->size(); ++index) {
        shape += (index > 0 ? "," : "") + TileShape((*tiles.children)[index]);
    }
    return shape + ")";
}

TileMeta Split(const std::string& split, const std::vector<std::string>& panes) {
    std::vector<TileMeta> children;
    for (const auto& pane : panes) {
        children.push_back({.pane = pane});
    }
    return TileMeta{.split = split, .children = std::move(children)};
}

struct TerminalRig {
    TestSupport::ScopedEnvironment shell{"SHELL"};
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> mails;
    TerminalActions actions{state, callbacks};

    std::shared_ptr<Flowgraph> flowgraph(const std::string& id) {
        auto graph = std::make_shared<Flowgraph>();
        state.flowgraph.items[id] = graph;
        state.flowgraph.restoredLayouts.insert(id);
        return graph;
    }

    void place(const std::string& flowgraph, TileMeta tiles) {
        LayoutActions layout(state.flowgraph);
        REQUIRE(layout.handle(MailSetFlowgraphLayout{.flowgraph = flowgraph, .tiles = std::move(tiles)}) ==
                Result::SUCCESS);
    }

    void settle() {
        actions.reconcile();
    }

    TerminalRig() {
        REQUIRE(shell.set(std::string("/bin/sh")));
        callbacks.enqueueMail = [this](Mail&& mail) {
            mails.push_back(std::move(mail));
        };
        callbacks.notify = [](Sakura::ToastType, I32, const std::string&) {};
    }

    ~TerminalRig() {
        actions.shutdown();
    }
};

}  // namespace

TEST_CASE("Terminal sessions render shell output through the emulator", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    REQUIRE(session.running());
    CHECK(session.screen().columns == 80);
    CHECK(session.screen().rows == 24);

    const U64 revision = session.revision();
    session.text("printf '\\033[31mred\\033[0m plain\\n'\r");
    REQUIRE(Shows(session, "red plain"));
    CHECK(session.revision() > revision);

    bool colored = false;
    for (const auto& row : session.screen().lines) {
        for (const auto& span : row.spans) {
            if (span.text == "red" && span.color.r > 0.5f && span.color.g < 0.5f) {
                colored = true;
            }
        }
    }
    CHECK(colored);
    CHECK(session.screen().cursor.visible);

    session.resize({.columns = 50, .rows = 12, .cellWidth = 8.0f, .cellHeight = 16.0f});
    CHECK(session.screen().columns == 50);
    CHECK(session.screen().rows == 12);
    CHECK(session.screen().lines.size() == 12);
    session.text("stty size\r");
    REQUIRE(Shows(session, "12 50"));

    session.text("exit\r");
    CHECK(PollUntil(session, [&] { return !session.running(); }));
}

TEST_CASE("Terminal sessions follow cursor moves that change no cells", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("printf '\\033[2J\\033[3;1Hstill'; sleep 1; printf '\\033[3;3H'; sleep 1; "
                 "printf '\\033[3;9H'; sleep 30\r");
    REQUIRE(PollUntil(session, [&] {
        const auto& cursor = session.screen().cursor;
        return ScreenText(session).find("still") != std::string::npos && cursor.row == 2 && cursor.column == 5;
    }));
    CHECK(PollUntil(session, [&] { return session.screen().cursor.column == 2; }));
    CHECK(PollUntil(session, [&] { return session.screen().cursor.column == 8; }));
    CHECK(session.screen().cursor.row == 2);
}

TEST_CASE("Terminal sessions follow cursor shape changes that change no cells", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("printf '\\033[2J\\033[3;1Hsteady\\033[2 q'; sleep 1; printf '\\033[6 q'; sleep 30\r");
    REQUIRE(PollUntil(session, [&] {
        return ScreenText(session).find("steady") != std::string::npos &&
               session.screen().cursor.shape == Sakura::Terminal::CursorShape::Block;
    }));
    CHECK(PollUntil(session, [&] { return session.screen().cursor.shape == Sakura::Terminal::CursorShape::Bar; }));
}

TEST_CASE("Terminal sessions encode special keys and pastes for the shell", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));

    session.text("echo typed");
    session.key({.type = KeyEventType::Press, .key = KeyCode::Enter});
    REQUIRE(Shows(session, "\ntyped"));

    session.text("sleep 30");
    session.key({.type = KeyEventType::Press, .key = KeyCode::Enter});
    session.key({.type = KeyEventType::Press, .key = KeyCode::C, .modifiers = {.control = true}});
    session.text("echo after$((1+1))\r");
    REQUIRE(Shows(session, "after2"));

    session.key({.type = KeyEventType::Press, .key = KeyCode::A});
    session.paste("echo pasted\n");
    REQUIRE(Shows(session, "\npasted"));
}

TEST_CASE("Terminal sessions select words and copy them as plain text", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("clear; printf 'alpha beta gamma\\n'\r");
    REQUIRE(PollUntil(session, [&] { return RowText(session.screen().lines[0]).find("alpha beta") == 0; }));

    using Gesture = Sakura::Terminal::Gesture;
    session.select({.gesture = Gesture::Press, .column = 7, .row = 0, .clicks = 2});
    session.poll();
    CHECK(session.screen().selection == "beta");
    REQUIRE(session.screen().lines[0].selection.has_value());
    CHECK(session.screen().lines[0].selection->start == 6);
    CHECK(session.screen().lines[0].selection->end == 9);

    session.select({.gesture = Gesture::Press, .column = 0, .row = 0, .clicks = 1});
    CHECK(session.screen().selection.empty());
    session.select({.gesture = Gesture::Drag, .column = 10, .row = 0, .clicks = 1});
    CHECK(session.screen().selection == "alpha beta");
    session.select({.gesture = Gesture::Release, .column = 10, .row = 0, .clicks = 1});
    CHECK_FALSE(session.screen().selection.empty());

    session.text("x");
    CHECK(session.screen().selection.empty());
}

TEST_CASE("Terminal sessions scroll back through history", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("i=0; while [ $i -lt 60 ]; do echo line$i; i=$((i+1)); done\r");
    REQUIRE(Shows(session, "line59"));
    const U64 bottom = session.screen().scrollOffset;
    REQUIRE(session.screen().scrollTotal > session.screen().rows);
    REQUIRE(bottom > 0);

    session.scroll(-5, {0.0f, 0.0f});
    CHECK(session.screen().scrollOffset == bottom - 5);
    session.scrollTo(0);
    CHECK(session.screen().scrollOffset == 0);
    CHECK(ScreenText(session).find("line0") != std::string::npos);

    session.text("echo back\r");
    CHECK(session.screen().scrollOffset >= bottom);
}

TEST_CASE("Terminal sessions fill only explicit and reverse video backgrounds", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("clear; printf 'A\\033[41mB\\033[0mC\\033[7mD\\033[0mE\\n'\r");
    REQUIRE(PollUntil(session, [&] { return RowText(session.screen().lines[0]).find("ABCDE") == 0; }));

    const auto& backgrounds = session.screen().lines[0].backgrounds;
    REQUIRE(backgrounds.size() == 2);
    CHECK(backgrounds[0].column == 1);
    CHECK(backgrounds[0].width == 1);
    CHECK(backgrounds[0].color.r > 0.5f);
    CHECK(backgrounds[0].color.g < 0.5f);
    CHECK(backgrounds[1].column == 3);
    CHECK(backgrounds[1].width == 1);
    for (U64 row = 1; row < session.screen().lines.size(); ++row) {
        CHECK(session.screen().lines[row].backgrounds.empty());
    }
}

TEST_CASE("Terminal sessions encode modified keys and mouse clicks for full screen apps", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    const std::string alt = TerminalSession::AltSendsEscape() ? " 1b 62 1b 21" : "";
    const U64 bytes = TerminalSession::AltSendsEscape() ? 30 : 26;
    TerminalSession session;
    REQUIRE(session.start({
        .program = "/bin/sh",
        .arguments = {"-c", "printf '\\033[?1049h\\033[?1000h'; stty raw -echo; "
                            "printf 'ready'; head -c " + std::to_string(bytes) +
                            " | od -An -tx1 | tr -s ' '; sleep 2"},
        .directory = "/",
    }) == Result::SUCCESS);
    session.resize({.columns = 60, .rows = 20, .cellWidth = 8.0f, .cellHeight = 16.0f});
    REQUIRE(Shows(session, "ready"));
    REQUIRE(PollUntil(session, [&] { return session.screen().mouseTracking; }));

    session.key({.type = KeyEventType::Press, .key = KeyCode::UpArrow});
    session.key({.type = KeyEventType::Press, .key = KeyCode::F1});
    session.key({.type = KeyEventType::Press, .key = KeyCode::A, .modifiers = {.control = true}});
    session.key({.type = KeyEventType::Press, .key = KeyCode::Minus, .modifiers = {.control = true, .shift = true}});
    session.key({.type = KeyEventType::Press, .key = KeyCode::B, .modifiers = {.alt = true}});
    session.key({.type = KeyEventType::Press, .key = KeyCode::Digit1, .modifiers = {.shift = true, .alt = true}});
    session.pointer({.type = MouseEventType::Click, .button = MouseButton::Left, .position = {20.0f, 56.0f}});
    session.pointer({.type = MouseEventType::Cancel, .position = {0.0f, 0.0f}});
    session.pointer({.type = MouseEventType::Cancel, .position = {0.0f, 0.0f}});
    session.scroll(-1, {20.0f, 56.0f}, {.control = true});
    const auto compact = [](std::string text) {
        std::erase_if(text, [](char value) {
            return value == ' ' || value == '\n';
        });
        return text;
    };
    const std::string expected = compact("1b 5b 41 1b 4f 50 01 1f" + alt +
                                         " 1b 5b 4d 20 23 24 1b 5b 4d 23 23 24 1b 5b 4d 70 23 24");
    REQUIRE(PollUntil(session, [&] { return compact(ScreenText(session)).find(expected) != std::string::npos; }));
}

TEST_CASE("Terminal sessions report TERM and resize the pseudo terminal", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("echo term=$TERM\r");
    REQUIRE(Shows(session, "term=xterm-256color"));
    session.resize({.columns = 72, .rows = 18, .cellWidth = 9.0f, .cellHeight = 18.0f});
    session.text("stty size\r");
    REQUIRE(Shows(session, "18 72"));
    session.resize({.columns = 41, .rows = 11, .cellWidth = 9.0f, .cellHeight = 18.0f});
    session.text("stty size\r");
    REQUIRE(Shows(session, "11 41"));
}

TEST_CASE("Terminal sessions turn wheel scrolls into arrows on the alternate screen", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(session.start({
        .program = "/bin/sh",
        .arguments = {"-c", "printf '\\033[?1049h'; stty raw -echo; printf 'ready'; head -c 3 | od -An -tx1 | tr -s ' '; sleep 2"},
        .directory = "/",
    }) == Result::SUCCESS);
    REQUIRE(Shows(session, "ready"));
    const U64 offset = session.screen().scrollOffset;
    session.scroll(-1, {0.0f, 0.0f});
    REQUIRE(Shows(session, "1b 5b 41"));
    CHECK(session.screen().scrollOffset == offset);
}

TEST_CASE("Terminal sessions deliver fast keystrokes in separate reads", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(session.start({
        .program = "/bin/sh",
        .arguments = {"-c", "stty raw -echo; printf 'ready'; dd bs=64 count=1 2>/dev/null | od -An -c | tr -s ' '; "
                            "printf 'done'; sleep 2"},
        .directory = "/",
    }) == Result::SUCCESS);
    REQUIRE(Shows(session, "ready"));
    session.text("a");
    session.text("b");
    session.text("c");
    REQUIRE(Shows(session, "done"));
    const std::string screen = ScreenText(session);
    CHECK(screen.find(" a") != std::string::npos);
    CHECK(screen.find(" a b") == std::string::npos);
}

TEST_CASE("Terminal sessions pace a burst of keystrokes in separate reads", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    const std::string reader =
        "my ($s, @r); for (1..4) { sysread(STDIN, my $b, 64); my $t = time; $s //= $t; "
        "push @r, sprintf('%s=%d', $b, ($t - $s) * 1e6); } print join(' ', @r), ' done'; sleep 5";
    TerminalSession session;
    REQUIRE(session.start({
        .program = "/bin/sh",
        .arguments = {"-c", "command -v perl >/dev/null 2>&1 || { printf 'noperl'; sleep 5; exit; }; "
                            "stty raw -echo; printf 'ready'; exec perl -MTime::HiRes=time -e \"$1\"",
                      "sh", reader},
        .directory = "/",
    }) == Result::SUCCESS);
    REQUIRE(PollUntil(session, [&] {
        const std::string screen = ScreenText(session);
        return screen.find("ready") != std::string::npos || screen.find("noperl") != std::string::npos;
    }));
    if (ScreenText(session).find("noperl") != std::string::npos) {
        SKIP("Perl is not available to time the reads.");
    }

    session.text("a");
    session.text("b");
    session.text("c");
    session.text("d");
    REQUIRE(Shows(session, "done"));

    const std::string screen = ScreenText(session);
    const auto begin = screen.find("a=");
    REQUIRE(begin != std::string::npos);
    std::istringstream records(screen.substr(begin, screen.find("done") - begin));
    std::vector<std::string> keys;
    std::vector<I64> times;
    std::string record;
    while (records >> record) {
        const auto separator = record.find('=');
        REQUIRE(separator != std::string::npos);
        keys.push_back(record.substr(0, separator));
        times.push_back(std::stoll(record.substr(separator + 1)));
    }
    REQUIRE(keys == std::vector<std::string>{"a", "b", "c", "d"});
    for (U64 index = 1; index < times.size(); ++index) {
        CHECK(times[index] > times[index - 1]);
    }
    CHECK(times.back() - times.front() >= 3 * 8000 / 2);
}

TEST_CASE("Terminal sessions stop promptly while the shell floods output", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    for (const bool polling : {true, false}) {
        TerminalSession session;
        REQUIRE(session.start({.program = "/bin/sh", .arguments = {"-c", "yes"}, .directory = "/"}) ==
                Result::SUCCESS);
        REQUIRE(Shows(session, "y"));
        const auto load = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
        while (std::chrono::steady_clock::now() < load) {
            if (polling) {
                session.poll();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        session.text("x");
        const auto begin = std::chrono::steady_clock::now();
        session.stop();
        CHECK(std::chrono::steady_clock::now() - begin < std::chrono::seconds(2));
        CHECK_FALSE(session.running());
        session.poll();
        CHECK_FALSE(session.running());
    }
}

TEST_CASE("Terminal sessions hold the screen during synchronized output", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(session.start({
        .program = "/bin/sh",
        .arguments = {"-c", "printf 'before\\n'; sleep 0.5; printf '\\033[?2026hduring\\n'; sleep 0.5; "
                            "printf 'after\\033[?2026l\\n'; sleep 2"},
        .directory = "/",
    }) == Result::SUCCESS);
    REQUIRE(Shows(session, "before"));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(800);
    while (std::chrono::steady_clock::now() < deadline) {
        session.poll();
        const std::string screen = ScreenText(session);
        CHECK((screen.find("during") == std::string::npos || screen.find("after") != std::string::npos));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(Shows(session, "after"));
    CHECK(ScreenText(session).find("during") != std::string::npos);
}

TEST_CASE("Terminal sessions show the frame finished right before a new synchronized hold", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(session.start({
        .program = "/bin/sh",
        .arguments = {"-c", "printf 'before\\n'; sleep 0.5; "
                            "printf '\\033[?2026hframe\\n\\033[?2026l\\033[?2026hpartial\\n'; sleep 0.6; "
                            "printf '\\033[?2026l'; sleep 2"},
        .directory = "/",
    }) == Result::SUCCESS);
    REQUIRE(Shows(session, "before"));
    REQUIRE(Shows(session, "frame"));
    CHECK(ScreenText(session).find("partial") == std::string::npos);
    REQUIRE(Shows(session, "partial"));
}

TEST_CASE("Opening a terminal gives it a new focused tab that starts in the flowgraph folder",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    const auto root = std::filesystem::temp_directory_path() / ("cyberether-terminal-" + Random::HexToken());
    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(graph->exportToFile(Platform::PathToUtf8(root / "graph.yml")) == Result::SUCCESS);
    rig.state.flowgraph.items["graph"] = graph;
    rig.state.interface.focusedFlowgraph = "graph";
    auto& terminals = rig.state.terminal;
    REQUIRE(rig.state.activeFlowgraph() == std::optional<std::string>("graph"));

    REQUIRE(rig.actions.handle(MailOpenTerminal{.flowgraph = "graph"}) == Result::SUCCESS);
    REQUIRE(terminals.items.size() == 1);
    REQUIRE(terminals.items.contains(1));
    CHECK(terminals.items.at(1).directory == Platform::PathToUtf8(root));
    REQUIRE(terminals.tabs.size() == 1);
    CHECK(TileShape(terminals.tabs.at(1)) == "terminal:1");
    CHECK(terminals.focused == std::optional<U64>(1));
    CHECK(terminals.requested == std::optional<U64>(1));
    CHECK(terminals.requests == 1);
    CHECK_FALSE(rig.state.interface.focusedTab.has_value());
    CHECK_FALSE(rig.state.activeFlowgraph().has_value());
    CHECK(rig.state.workspaceTabs() == 2);

    rig.settle();
    CHECK(terminals.items.at(1).tab == std::optional<U64>(1));
    CHECK_FALSE(terminals.items.at(1).flowgraph.has_value());

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(terminals.tabs.size() == 2);
    CHECK(TileShape(terminals.tabs.at(2)) == "terminal:2");
    CHECK(terminals.items.at(2).directory.empty());
    CHECK(terminals.focused == std::optional<U64>(2));
    CHECK(terminals.requested == std::optional<U64>(2));
    CHECK(terminals.requests == 2);

    REQUIRE(rig.actions.handle(MailOpenTerminal{.flowgraph = "ghost"}) == Result::SUCCESS);
    CHECK(TileShape(terminals.tabs.at(3)) == "terminal:3");
    CHECK(terminals.items.at(3).directory.empty());
    rig.settle();
    CHECK(terminals.items.at(2).tab == std::optional<U64>(2));
    CHECK(terminals.items.at(3).tab == std::optional<U64>(3));
    CHECK(terminals.focused == std::optional<U64>(3));

    REQUIRE(rig.actions.handle(MailFocusTerminalTab{.tab = 1}) == Result::SUCCESS);
    CHECK(terminals.focused == std::optional<U64>(1));
    REQUIRE(rig.actions.handle(MailFocusTerminalTab{.tab = 9}) == Result::SUCCESS);
    CHECK(terminals.focused == std::optional<U64>(1));
    CHECK(terminals.requested == std::optional<U64>(3));
    CHECK_FALSE(rig.state.activeFlowgraph().has_value());

    WorkbenchActions::Focus(rig.state, std::nullopt);
    CHECK_FALSE(terminals.focused.has_value());
    CHECK(rig.state.activeFlowgraph() == std::optional<std::string>("graph"));

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("Splitting a terminal in a tab places the new one beside or below it in the same folder",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    auto& terminals = rig.state.terminal;
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    terminals.items.at(1).directory = "/tmp";
    rig.settle();

    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 1}) == Result::SUCCESS);
    REQUIRE(terminals.items.contains(2));
    CHECK(terminals.items.at(2).directory == "/tmp");
    REQUIRE(terminals.tabs.size() == 1);
    CHECK(TileShape(terminals.tabs.at(1)) == "row(terminal:1,terminal:2)");
    CHECK(terminals.requests == 1);
    CHECK(terminals.focused == std::optional<U64>(1));

    rig.settle();
    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 2, .stacked = true}) == Result::SUCCESS);
    CHECK(terminals.items.at(3).directory == "/tmp");
    CHECK(TileShape(terminals.tabs.at(1)) == "row(terminal:1,column(terminal:2,terminal:3))");

    rig.settle();
    REQUIRE(terminals.tabs.size() == 1);
    for (const U64 id : {1, 2, 3}) {
        CHECK(terminals.items.at(id).tab == std::optional<U64>(1));
    }

    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 9}) == Result::SUCCESS);
    REQUIRE(terminals.tabs.size() == 2);
    CHECK(TileShape(terminals.tabs.at(2)) == "terminal:4");
    CHECK(terminals.items.at(4).directory.empty());
    CHECK(terminals.focused == std::optional<U64>(2));
    CHECK(terminals.requests == 2);
}

TEST_CASE("Splitting a terminal tiled in a flowgraph splits it inside that layout and keeps terminals out of saved metadata",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    auto graph = rig.flowgraph("graph");
    auto& terminals = rig.state.terminal;
    const auto saved = [&] {
        TileMeta tiles;
        REQUIRE(graph->metadata().get("layout", tiles) == Result::SUCCESS);
        return TileShape(tiles);
    };

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    terminals.items.at(1).directory = "/tmp";
    rig.place("graph", Split("row", {"flowgraph", "terminal:1"}));
    CHECK(saved() == "flowgraph");
    rig.settle();
    CHECK(terminals.items.at(1).flowgraph == std::optional<std::string>("graph"));
    CHECK_FALSE(terminals.items.at(1).tab.has_value());
    CHECK(terminals.tabs.empty());
    CHECK_FALSE(terminals.focused.has_value());
    CHECK(TileShape(rig.state.flowgraph.layouts.at("graph")) == "row(flowgraph,terminal:1)");

    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 1}) == Result::SUCCESS);
    CHECK(TileShape(rig.state.flowgraph.layouts.at("graph")) == "row(flowgraph,terminal:1,terminal:2)");
    CHECK(terminals.items.at(2).directory == "/tmp");
    rig.settle();
    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 2, .stacked = true}) == Result::SUCCESS);
    CHECK(TileShape(rig.state.flowgraph.layouts.at("graph")) ==
          "row(flowgraph,terminal:1,column(terminal:2,terminal:3))");
    CHECK(terminals.tabs.empty());
    CHECK(terminals.requests == 1);

    rig.settle();
    for (const U64 id : {1, 2, 3}) {
        CHECK(terminals.items.at(id).flowgraph == std::optional<std::string>("graph"));
    }
    CHECK(terminals.tabs.empty());

    rig.place("graph", rig.state.flowgraph.layouts.at("graph"));
    CHECK(TileShape(rig.state.flowgraph.layouts.at("graph")) ==
          "row(flowgraph,terminal:1,column(terminal:2,terminal:3))");
    CHECK(saved() == "flowgraph");
}

TEST_CASE("Settling lets the first layout holding a terminal claim it and strips unknown and duplicate keys",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    rig.flowgraph("graph");
    rig.flowgraph("other");
    auto& terminals = rig.state.terminal;
    auto& layouts = rig.state.flowgraph.layouts;

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(terminals.focused == std::optional<U64>(2));

    rig.place("graph", Split("row", {"flowgraph", "terminal:2", "terminal:7"}));
    rig.settle();
    CHECK(TileShape(layouts.at("graph")) == "row(flowgraph,terminal:2)");
    CHECK(terminals.items.at(2).flowgraph == std::optional<std::string>("graph"));
    CHECK_FALSE(terminals.items.at(2).tab.has_value());
    REQUIRE(terminals.tabs.size() == 1);
    CHECK(TileShape(terminals.tabs.at(1)) == "terminal:1");
    CHECK(terminals.items.at(1).tab == std::optional<U64>(1));
    CHECK_FALSE(terminals.focused.has_value());

    REQUIRE(rig.actions.handle(MailSetTerminalTabLayout{
        .tab = 1,
        .tiles = Split("row", {"terminal:1", "terminal:2", "terminal:8"}),
    }) == Result::SUCCESS);
    rig.settle();
    CHECK(TileShape(terminals.tabs.at(1)) == "terminal:1");
    CHECK(TileShape(layouts.at("graph")) == "row(flowgraph,terminal:2)");
    CHECK(terminals.items.at(2).flowgraph == std::optional<std::string>("graph"));

    rig.place("other", Split("row", {"flowgraph", "terminal:2"}));
    rig.settle();
    U64 holders = 0;
    for (const auto& [id, tiles] : layouts) {
        holders += ContainsTilePane(tiles, MakeTerminalPaneKey(2)) ? 1 : 0;
    }
    CHECK(holders == 1);
    REQUIRE(terminals.items.at(2).flowgraph.has_value());
    CHECK(ContainsTilePane(layouts.at(*terminals.items.at(2).flowgraph), MakeTerminalPaneKey(2)));
    CHECK(terminals.items.size() == 2);
    CHECK(terminals.tabs.size() == 1);
}

TEST_CASE("Terminals that no layout claims get a new focused tab and focus leaves a tab that disappears",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    rig.flowgraph("graph");
    auto& terminals = rig.state.terminal;
    auto& layouts = rig.state.flowgraph.layouts;

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.place("graph", Split("row", {"flowgraph", "terminal:1"}));
    rig.settle();
    CHECK(terminals.tabs.empty());
    CHECK_FALSE(terminals.focused.has_value());
    CHECK(terminals.requests == 1);

    rig.place("graph", Split("row", {"flowgraph"}));
    rig.settle();
    CHECK_FALSE(terminals.items.at(1).flowgraph.has_value());
    CHECK(terminals.items.at(1).tab == std::optional<U64>(2));
    REQUIRE(terminals.tabs.size() == 1);
    CHECK(TileShape(terminals.tabs.at(2)) == "terminal:1");
    CHECK(terminals.focused == std::optional<U64>(2));
    CHECK(terminals.requested == std::optional<U64>(2));
    CHECK(terminals.requests == 2);

    rig.place("graph", Split("row", {"flowgraph", "terminal:1"}));
    rig.settle();
    CHECK(terminals.tabs.empty());
    CHECK_FALSE(terminals.focused.has_value());
    CHECK(terminals.requested == std::optional<U64>(2));

    rig.state.flowgraph.items.erase("graph");
    rig.settle();
    CHECK(TileShape(layouts.at("graph")) == "flowgraph");
    CHECK_FALSE(terminals.items.at(1).flowgraph.has_value());
    CHECK(terminals.items.at(1).tab == std::optional<U64>(3));
    CHECK(terminals.focused == std::optional<U64>(3));
    CHECK(terminals.requests == 3);
}

TEST_CASE("Settling splices a terminal tab dropped into a flowgraph or another tab and strips a tab's own key",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    auto graph = rig.flowgraph("graph");
    auto& terminals = rig.state.terminal;
    auto& layouts = rig.state.flowgraph.layouts;

    for (U64 index = 0; index < 3; ++index) {
        REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    }
    rig.settle();
    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 2}) == Result::SUCCESS);
    rig.settle();
    REQUIRE(TileShape(terminals.tabs.at(2)) == "row(terminal:2,terminal:4)");

    rig.place("graph", Split("row", {"flowgraph", "terminals:2", "terminals:99"}));
    TileMeta saved;
    REQUIRE(graph->metadata().get("layout", saved) == Result::SUCCESS);
    CHECK(TileShape(saved) == "flowgraph");
    rig.settle();
    CHECK(TileShape(layouts.at("graph")) == "row(flowgraph,row(terminal:2,terminal:4))");
    CHECK_FALSE(terminals.tabs.contains(2));
    CHECK(terminals.tabs.size() == 2);
    CHECK(terminals.items.at(2).flowgraph == std::optional<std::string>("graph"));
    CHECK(terminals.items.at(4).flowgraph == std::optional<std::string>("graph"));

    REQUIRE(rig.actions.handle(MailSetTerminalTabLayout{
        .tab = 1,
        .tiles = Split("column", {"terminals:3", "terminal:1"}),
    }) == Result::SUCCESS);
    rig.settle();
    REQUIRE(terminals.tabs.size() == 1);
    CHECK(TileShape(terminals.tabs.at(1)) == "column(terminal:3,terminal:1)");
    CHECK(terminals.items.at(3).tab == std::optional<U64>(1));
    CHECK(terminals.items.at(1).tab == std::optional<U64>(1));

    REQUIRE(rig.actions.handle(MailSetTerminalTabLayout{
        .tab = 1,
        .tiles = Split("column", {"terminal:3", "terminals:1", "terminal:1"}),
    }) == Result::SUCCESS);
    rig.settle();
    REQUIRE(terminals.tabs.size() == 1);
    CHECK(TileShape(terminals.tabs.at(1)) == "column(terminal:3,terminal:1)");
    CHECK(terminals.items.size() == 4);
}

TEST_CASE("Settling keeps terminals that sit before a spliced tab and keeps the first of repeated keys",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    auto graph = rig.flowgraph("graph");
    auto& terminals = rig.state.terminal;
    auto& layouts = rig.state.flowgraph.layouts;

    for (U64 index = 0; index < 3; ++index) {
        REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    }
    rig.settle();

    REQUIRE(rig.actions.handle(MailSetTerminalTabLayout{
        .tab = 1,
        .tiles = Split("row", {"terminal:1", "terminals:3"}),
    }) == Result::SUCCESS);
    rig.settle();
    CHECK(TileShape(terminals.tabs.at(1)) == "row(terminal:1,terminal:3)");
    CHECK_FALSE(terminals.tabs.contains(3));
    CHECK(terminals.tabs.size() == 2);
    CHECK(terminals.items.at(1).tab == std::optional<U64>(1));
    CHECK(terminals.items.at(3).tab == std::optional<U64>(1));

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.settle();
    REQUIRE(terminals.tabs.contains(4));
    rig.place("graph", Split("row", {"flowgraph", "terminal:2", "terminals:4"}));
    rig.settle();
    CHECK(TileShape(layouts.at("graph")) == "row(flowgraph,terminal:2,terminal:4)");
    CHECK(terminals.items.at(2).flowgraph == std::optional<std::string>("graph"));
    CHECK(terminals.items.at(4).flowgraph == std::optional<std::string>("graph"));
    CHECK(terminals.tabs.size() == 1);
    CHECK(terminals.tabs.contains(1));

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.settle();
    REQUIRE(terminals.tabs.contains(5));
    REQUIRE(rig.actions.handle(MailSetTerminalTabLayout{
        .tab = 5,
        .tiles = Split("row", {"terminals:1", "terminal:5"}),
    }) == Result::SUCCESS);
    rig.settle();
    REQUIRE(terminals.tabs.size() == 1);
    CHECK(TileShape(terminals.tabs.at(5)) == "row(row(terminal:1,terminal:3),terminal:5)");
    CHECK(terminals.items.at(1).tab == std::optional<U64>(5));
    CHECK(terminals.items.at(3).tab == std::optional<U64>(5));

    rig.place("graph", Split("row", {"flowgraph", "terminal:2", "terminal:2", "terminal:4"}));
    rig.settle();
    CHECK(TileShape(layouts.at("graph")) == "row(flowgraph,terminal:2,terminal:4)");
    CHECK(terminals.items.at(2).flowgraph == std::optional<std::string>("graph"));
    CHECK(terminals.tabs.size() == 1);
    CHECK(terminals.items.size() == 5);
}

TEST_CASE("Closing a terminal removes its key and closing a tab closes its terminals",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    rig.flowgraph("graph");
    auto& terminals = rig.state.terminal;
    auto& layouts = rig.state.flowgraph.layouts;

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.settle();
    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 1}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.place("graph", Split("row", {"flowgraph", "terminal:4"}));
    rig.settle();
    REQUIRE(TileShape(terminals.tabs.at(1)) == "row(terminal:1,terminal:2)");
    REQUIRE(TileShape(terminals.tabs.at(2)) == "terminal:3");
    REQUIRE(terminals.tabs.size() == 2);
    REQUIRE(terminals.items.at(4).flowgraph == std::optional<std::string>("graph"));

    REQUIRE(rig.actions.handle(MailCloseTerminal{.terminal = 2}) == Result::SUCCESS);
    CHECK_FALSE(terminals.items.contains(2));
    CHECK(TileShape(terminals.tabs.at(1)) == "terminal:1");

    REQUIRE(rig.actions.handle(MailCloseTerminal{.terminal = 4}) == Result::SUCCESS);
    CHECK_FALSE(terminals.items.contains(4));
    CHECK(TileShape(layouts.at("graph")) == "flowgraph");

    REQUIRE(rig.actions.handle(MailSetTerminalTabLayout{.tab = 9, .tiles = Split("row", {"terminal:3"})}) ==
            Result::SUCCESS);
    CHECK_FALSE(terminals.tabs.contains(9));

    REQUIRE(rig.actions.handle(MailFocusTerminalTab{.tab = 2}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailCloseTerminalTab{.tab = 2}) == Result::SUCCESS);
    CHECK_FALSE(terminals.items.contains(3));
    CHECK_FALSE(terminals.tabs.contains(2));
    CHECK_FALSE(terminals.focused.has_value());

    REQUIRE(rig.actions.handle(MailFocusTerminalTab{.tab = 1}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailCloseTerminal{.terminal = 1}) == Result::SUCCESS);
    CHECK(terminals.items.empty());
    CHECK(terminals.tabs.empty());
    CHECK_FALSE(terminals.focused.has_value());
    CHECK(rig.state.workspaceTabs() == 1);
}

TEST_CASE("Closing a terminal tab keeps a terminal that was dragged out of it in the same batch",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    auto& terminals = rig.state.terminal;

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 1}) == Result::SUCCESS);
    rig.settle();
    REQUIRE(TileShape(terminals.tabs.at(1)) == "row(terminal:1,terminal:2)");

    REQUIRE(rig.actions.handle(MailSetTerminalTabLayout{.tab = 1, .tiles = TileMeta{.pane = "terminal:2"}}) ==
            Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailCloseTerminalTab{.tab = 1}) == Result::SUCCESS);
    CHECK_FALSE(terminals.items.contains(2));
    REQUIRE(terminals.items.contains(1));
    CHECK_FALSE(terminals.tabs.contains(1));
    REQUIRE(terminals.tabs.size() == 1);
    const U64 tab = terminals.tabs.begin()->first;
    CHECK(TileShape(terminals.tabs.at(tab)) == "terminal:1");
    CHECK(terminals.items.at(1).tab == std::optional<U64>(tab));
    CHECK(terminals.focused == std::optional<U64>(tab));
}

TEST_CASE("Splitting a terminal side by side in a flowgraph column adds a column after it",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    rig.flowgraph("graph");
    auto& layouts = rig.state.flowgraph.layouts;

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.place("graph", TileMeta{
        .split = "row",
        .children = std::vector<TileMeta>{TileMeta{.pane = "flowgraph"},
                                          Split("column", {"terminal:1", "terminal:2"})},
    });
    rig.settle();
    REQUIRE(TileShape(layouts.at("graph")) == "row(flowgraph,column(terminal:1,terminal:2))");

    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 2}) == Result::SUCCESS);
    CHECK(TileShape(layouts.at("graph")) == "row(flowgraph,column(terminal:1,terminal:2),terminal:3)");
    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 1, .stacked = true}) == Result::SUCCESS);
    CHECK(TileShape(layouts.at("graph")) ==
          "row(flowgraph,column(column(terminal:1,terminal:4),terminal:2),terminal:3)");
    rig.settle();
    CHECK(rig.state.terminal.tabs.empty());
    CHECK(rig.state.terminal.items.at(3).flowgraph == std::optional<std::string>("graph"));
    CHECK(rig.state.terminal.items.at(4).flowgraph == std::optional<std::string>("graph"));

    CHECK(TileShape(SplitColumnPane(TileMeta{.pane = "a"}, "a", "b", false)) == "row(a,b)");
    CHECK(TileShape(SplitColumnPane(Split("column", {"a", "c"}), "c", "b", false)) == "row(column(a,c),b)");
    CHECK(TileShape(SplitColumnPane(Split("row", {"a", "c"}), "x", "b", false)) == "row(a,c)");
}

TEST_CASE("Activating a terminal tab asks its last focused terminal for keyboard focus",
          "[core][compositor][terminal][focus]") {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalRig rig;
    auto& terminals = rig.state.terminal;
    const auto focus = [&](U64 terminal) {
        REQUIRE(rig.actions.handle(MailTerminalInput{.terminal = terminal,
                                                     .input = MailTerminalInput::Focus{.value = true}}) ==
                Result::SUCCESS);
    };

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailOpenTerminal{.split = 1}) == Result::SUCCESS);
    rig.settle();

    REQUIRE(rig.actions.handle(MailFocusTerminalTab{.tab = 1}) == Result::SUCCESS);
    CHECK(terminals.items.at(1).focusRequests == 1);
    CHECK(terminals.items.at(2).focusRequests == 0);

    focus(2);
    REQUIRE(rig.actions.handle(MailFocusTerminalTab{.tab = 1}) == Result::SUCCESS);
    CHECK(terminals.items.at(1).focusRequests == 1);
    CHECK(terminals.items.at(2).focusRequests == 1);

    focus(1);
    REQUIRE(rig.actions.handle(MailTerminalInput{.terminal = 2, .input = MailTerminalInput::Focus{.value = false}}) ==
            Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailFocusTerminalTab{.tab = 1}) == Result::SUCCESS);
    CHECK(terminals.items.at(1).focusRequests == 2);
    CHECK(terminals.items.at(2).focusRequests == 1);

    REQUIRE(rig.actions.handle(MailFocusTerminalTab{.tab = 7}) == Result::SUCCESS);
    CHECK(terminals.items.at(1).focusRequests == 2);
    CHECK(terminals.focused == std::optional<U64>(1));
}

TEST_CASE("Opening a terminal as a flowgraph tile appends it beside the editor without a tab",
          "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    auto graph = rig.flowgraph("graph");
    auto& terminals = rig.state.terminal;
    auto& layouts = rig.state.flowgraph.layouts;

    REQUIRE(rig.actions.handle(MailOpenTerminal{.flowgraph = "graph", .tile = true}) == Result::SUCCESS);
    CHECK(TileShape(layouts.at("graph")) == "row(flowgraph,terminal:1)");
    CHECK(terminals.tabs.empty());
    CHECK(terminals.requests == 0);
    REQUIRE(rig.actions.handle(MailOpenTerminal{.flowgraph = "graph", .tile = true}) == Result::SUCCESS);
    CHECK(TileShape(layouts.at("graph")) == "row(flowgraph,terminal:1,terminal:2)");
    rig.settle();
    CHECK(terminals.items.at(1).flowgraph == std::optional<std::string>("graph"));
    CHECK(terminals.items.at(2).flowgraph == std::optional<std::string>("graph"));
    CHECK(terminals.tabs.empty());
    CHECK_FALSE(graph->metadata().has("layout"));

    REQUIRE(rig.actions.handle(MailOpenTerminal{.flowgraph = "missing", .tile = true}) == Result::SUCCESS);
    CHECK(terminals.tabs.size() == 1);
    CHECK(terminals.items.at(3).tab.has_value());
}

TEST_CASE("Closing a flowgraph moves its terminals together into one new tab", "[core][compositor][terminal][layout]") {
    TerminalRig rig;
    rig.flowgraph("graph");
    auto& terminals = rig.state.terminal;
    auto& layouts = rig.state.flowgraph.layouts;

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.place("graph", TileMeta{
        .split = "row",
        .children = std::vector<TileMeta>{TileMeta{.pane = "flowgraph"},
                                          Split("column", {"terminal:1", "terminal:2"})},
    });
    rig.settle();
    REQUIRE(terminals.tabs.size() == 1);
    const U64 requests = terminals.requests;

    rig.state.flowgraph.items.erase("graph");
    rig.settle();
    CHECK(TileShape(layouts.at("graph")) == "flowgraph");
    REQUIRE(terminals.tabs.size() == 2);
    const U64 tab = terminals.tabs.rbegin()->first;
    CHECK(TileShape(terminals.tabs.at(tab)) == "column(terminal:1,terminal:2)");
    CHECK(terminals.items.at(1).tab == std::optional<U64>(tab));
    CHECK(terminals.items.at(2).tab == std::optional<U64>(tab));
    CHECK(terminals.focused == std::optional<U64>(tab));
    CHECK(terminals.requested == std::optional<U64>(tab));
    CHECK(terminals.requests == requests + 1);
}

TEST_CASE("Opening a terminal from a new start tab replaces it like a new flowgraph",
          "[core][compositor][terminal][start-tab]") {
    TerminalRig rig;
    WorkbenchActions workbench(rig.state, rig.callbacks);
    workbench.settleStartTab();
    CHECK(rig.state.startTabActive());

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    workbench.settleStartTab();
    CHECK_FALSE(rig.state.startTabShown());
    CHECK_FALSE(rig.state.tabsHidden());
    CHECK(rig.state.terminal.focused == std::optional<U64>(1));

    rig.flowgraph("graph");
    workbench.handle(MailOpenStartTab{});
    CHECK(rig.state.interface.startTabBaseline == 2);
    workbench.settleStartTab();
    CHECK(rig.state.interface.startTab);
    CHECK(rig.state.startTabActive());
    CHECK_FALSE(rig.state.terminal.focused.has_value());

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    workbench.settleStartTab();
    CHECK_FALSE(rig.state.interface.startTab);
    CHECK_FALSE(rig.state.startTabActive());
    CHECK(rig.state.terminal.focused == std::optional<U64>(2));

    rig.settle();
    REQUIRE(rig.actions.handle(MailCloseTerminalTab{.tab = 1}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailCloseTerminalTab{.tab = 2}) == Result::SUCCESS);
    rig.state.flowgraph.items.erase("graph");
    workbench.settleStartTab();
    CHECK(rig.state.startTabActive());
}

TEST_CASE("Terminal actions run a session per open terminal, publish each screen, and route input by id", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalRig rig;
    auto& items = rig.state.terminal.items;
    const auto settle = [&](const std::function<bool()>& done) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            rig.actions.reconcile();
            if (done()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    };
    const auto text = [&](U64 id) {
        std::string value;
        for (const auto& row : items.at(id).screen.lines) {
            value += RowText(row);
            value += '\n';
        }
        return value;
    };

    rig.actions.reconcile();
    CHECK(rig.actions.sessions.empty());
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.actions.reconcile();
    REQUIRE(rig.actions.sessions.size() == 2);
    CHECK(items.at(1).revision > 0);
    CHECK(items.at(2).revision > 0);
    const U64 first = items.at(1).revision;

    REQUIRE(rig.actions.handle(MailTerminalInput{.terminal = 1, .input = MailTerminalInput::Text{"echo publ$((1+1))ished\r"}}) ==
            Result::SUCCESS);
    REQUIRE(settle([&] { return text(1).find("publ2ished") != std::string::npos; }));
    CHECK(items.at(1).revision > first);
    CHECK(text(2).find("publ2ished") == std::string::npos);

    REQUIRE(rig.actions.handle(MailTerminalInput{
        .terminal = 2,
        .input = Sakura::Terminal::Grid{.columns = 50, .rows = 10, .cellWidth = 8.0f, .cellHeight = 16.0f},
    }) == Result::SUCCESS);
    CHECK(items.at(2).screen.columns == 50);
    CHECK(items.at(2).screen.rows == 10);
    CHECK(items.at(1).screen.columns == 80);

    REQUIRE(rig.actions.handle(MailTerminalInput{.terminal = 9, .input = MailTerminalInput::Text{"x"}}) ==
            Result::SUCCESS);
    CHECK(rig.mails.empty());

    REQUIRE(rig.actions.handle(MailTerminalInput{.terminal = 1, .input = MailTerminalInput::Text{"exit\r"}}) ==
            Result::SUCCESS);
    REQUIRE(settle([&] { return !rig.mails.empty(); }));
    REQUIRE(rig.mails.size() == 1);
    const auto* closed = std::get_if<MailCloseTerminal>(&rig.mails.front());
    REQUIRE(closed != nullptr);
    CHECK(closed->terminal == 1);

    REQUIRE(rig.actions.handle(*closed) == Result::SUCCESS);
    CHECK_FALSE(items.contains(1));
    rig.actions.reconcile();
    CHECK(rig.actions.sessions.size() == 1);
    CHECK(rig.actions.sessions.contains(2));

    REQUIRE(rig.actions.handle(MailCloseTerminal{.terminal = 2}) == Result::SUCCESS);
    rig.actions.reconcile();
    CHECK(rig.actions.sessions.empty());

    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.actions.reconcile();
    REQUIRE(rig.actions.sessions.size() == 1);
    CHECK(rig.actions.sessions.contains(3));
    CHECK(items.at(3).revision > 0);
    rig.actions.shutdown();
    CHECK(rig.actions.sessions.empty());
}

TEST_CASE("Terminal sessions keep long graphemes without overflowing the cell buffer", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    std::string marks;
    for (int index = 0; index < 32; ++index) {
        marks += "\\314\\201";
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("clear; printf 'A" + marks + "B\\n'\r");
    REQUIRE(PollUntil(session, [&] { return RowText(session.screen().lines[0]).find('B') != std::string::npos; }));
    CHECK(RowText(session.screen().lines[0]).rfind("A", 0) == 0);
}

TEST_CASE("Terminal zoom steps clamp to their limits and zero resets them", "[core][compositor][terminal][layout]") {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalRig rig;
    rig.shell.set("/bin/sh");
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.actions.reconcile();
    REQUIRE(rig.actions.sessions.contains(1));
    const auto zoom = [&](I32 step) {
        REQUIRE(rig.actions.handle(MailTerminalInput{.terminal = 1, .input = MailTerminalInput::Zoom{.step = step}}) ==
                Result::SUCCESS);
        return rig.state.terminal.items.at(1).zoom;
    };
    CHECK(zoom(1) == 1);
    CHECK(zoom(-1) == 0);
    for (I32 index = 0; index < 40; ++index) {
        zoom(1);
    }
    CHECK(zoom(1) == TerminalActions::kMaxZoom);
    for (I32 index = 0; index < 60; ++index) {
        zoom(-1);
    }
    CHECK(zoom(-1) == TerminalActions::kMinZoom);
    CHECK(zoom(0) == 0);
}

TEST_CASE("Terminal sessions publish hyperlinks and plain URLs per row", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("printf '\\033[2J\\033[H\\033]8;;https://example.org/a\\033\\\\label\\033]8;;\\033\\\\ "
                 "see (https://cyberether.org/docs). x'; sleep 30\r");
    const auto links = [&]() -> std::vector<Sakura::Terminal::Link> {
        return session.screen().lines.empty() ? std::vector<Sakura::Terminal::Link>{} : session.screen().lines[0].links;
    };
    REQUIRE(PollUntil(session, [&] { return links().size() == 2; }));
    CHECK(links()[0].uri == "https://example.org/a");
    CHECK(links()[0].column == 0);
    CHECK(links()[0].width == 5);
    CHECK(links()[1].uri == "https://cyberether.org/docs");
    CHECK(links()[1].column == 11);
    CHECK(links()[1].width == 27);
}

TEST_CASE("Terminal sessions link a URL that wraps across rows to its whole destination", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.resize({.columns = 20, .rows = 10, .cellWidth = 8.0f, .cellHeight = 16.0f});
    session.text("printf '\\033[2J\\033[Hhttps://example.org/a/long/path end'; sleep 30\r");
    const auto row = [&](U16 index) -> std::vector<Sakura::Terminal::Link> {
        return session.screen().lines.size() > index ? session.screen().lines[index].links
                                                     : std::vector<Sakura::Terminal::Link>{};
    };
    REQUIRE(PollUntil(session, [&] { return row(0).size() == 1 && row(1).size() == 1; }));
    CHECK(row(0)[0].uri == "https://example.org/a/long/path");
    CHECK(row(0)[0].column == 0);
    CHECK(row(0)[0].width == 20);
    CHECK(row(1)[0].uri == "https://example.org/a/long/path");
    CHECK(row(1)[0].column == 0);
    CHECK(row(1)[0].width == 11);
}

TEST_CASE("Terminal sessions skip a wrapped URL that continues below the visible rows", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.resize({.columns = 20, .rows = 6, .cellWidth = 8.0f, .cellHeight = 16.0f});
    session.text("clear; i=0; while [ $i -lt 30 ]; do echo line$i; i=$((i+1)); done; "
                 "printf 'https://example.org/a/long/path\\n'; printf 'tail'; sleep 30\r");
    REQUIRE(Shows(session, "tail"));
    const auto bottomLink = [&] {
        for (const auto& row : session.screen().lines) {
            for (const auto& link : row.links) {
                if (link.uri.find("example.org") != std::string::npos) {
                    return link.uri;
                }
            }
        }
        return std::string();
    };
    REQUIRE(bottomLink() == "https://example.org/a/long/path");
    const auto bottom = [&] {
        const auto& lines = session.screen().lines;
        return lines.empty() ? std::string() : RowText(lines.back());
    };
    for (U64 step = 0; step < 10 && bottom().rfind("https://", 0) != 0; ++step) {
        session.scroll(-1, {10.0f, 10.0f}, {});
        session.poll();
    }
    REQUIRE(bottom().rfind("https://example.org/", 0) == 0);
    REQUIRE(bottom().find("path") == std::string::npos);
    const auto& lines = session.screen().lines;
    bool truncated = false;
    for (const auto& row : lines) {
        for (const auto& link : row.links) {
            truncated |= link.uri != "https://example.org/a/long/path" && link.uri.find("example.org") != std::string::npos;
        }
    }
    CHECK_FALSE(truncated);
}

TEST_CASE("Terminal sessions keep the shown hyperlink while synchronized output holds a new one", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(session.start({
        .program = "/bin/sh",
        .arguments = {"-c", "printf '\\033]8;;https://old.example\\033\\\\label\\033]8;;\\033\\\\'; sleep 0.5; "
                            "printf '\\033[H\\033]8;;https://mid.example\\033\\\\label\\033]8;;\\033\\\\"
                            "\\033[?2026h\\033[H\\033]8;;https://new.example\\033\\\\label\\033]8;;\\033\\\\'; "
                            "sleep 0.8; printf '\\033[?2026l'; sleep 2"},
        .directory = "/",
    }) == Result::SUCCESS);
    const auto uri = [&]() -> std::string {
        const auto& lines = session.screen().lines;
        return lines.empty() || lines[0].links.empty() ? std::string() : lines[0].links[0].uri;
    };
    REQUIRE(PollUntil(session, [&] { return uri() == "https://old.example"; }));
    REQUIRE(PollUntil(session, [&] { return uri() != "https://old.example"; }));
    CHECK(uri() == "https://mid.example");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        session.poll();
        CHECK(uri() == "https://mid.example");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    REQUIRE(PollUntil(session, [&] { return uri() == "https://new.example"; }));
}

TEST_CASE("Terminal link opens forward only web, mail, and file links", "[core][compositor][terminal][layout]") {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalRig rig;
    rig.shell.set("/bin/sh");
    REQUIRE(rig.actions.handle(MailOpenTerminal{}) == Result::SUCCESS);
    rig.actions.reconcile();
    REQUIRE(rig.actions.sessions.contains(1));
    rig.mails.clear();
    for (const auto* uri : {"https://example.org", "HTTP://example.org", "mailto:a@b.c", "file:///tmp/x",
                            "javascript:alert(1)", "ssh://host", "https://"}) {
        REQUIRE(rig.actions.handle(MailTerminalInput{.terminal = 1, .input = MailTerminalInput::OpenLink{.uri = uri}}) ==
                Result::SUCCESS);
    }
    std::vector<std::string> opened;
    for (const auto& mail : rig.mails) {
        if (const auto* open = std::get_if<MailOpenUrl>(&mail)) {
            opened.push_back(open->url);
        }
    }
    CHECK(opened == std::vector<std::string>{"https://example.org", "HTTP://example.org", "mailto:a@b.c", "file:///tmp/x"});
}

TEST_CASE("Terminal sessions search the screen, step between matches, and clear the search", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("clear; printf 'alpha needle\\nbeta\\nNEEDLE gamma needle\\n'; sleep 30\r");
    REQUIRE(Shows(session, "gamma"));
    session.search("needle");
    const auto& screen = session.screen();
    CHECK(screen.matchCount == 3);
    REQUIRE(screen.matchIndex.has_value());
    const auto marked = [&] {
        U64 count = 0;
        U64 current = 0;
        for (const auto& row : screen.lines) {
            count += row.matches.size();
            current += row.match.has_value() ? 1 : 0;
        }
        return std::make_pair(count, current);
    };
    CHECK(marked().second == 1);
    CHECK(marked().first + marked().second >= 3);
    const U64 first = *screen.matchIndex;
    session.searchStep(true);
    REQUIRE(screen.matchIndex.has_value());
    CHECK(*screen.matchIndex == (first + 1) % 3);
    session.searchStep(false);
    CHECK(*screen.matchIndex == first);

    session.search("");
    CHECK(screen.matchCount == 0);
    CHECK_FALSE(screen.matchIndex.has_value());
    CHECK(marked() == std::make_pair(U64{0}, U64{0}));
}

TEST_CASE("Terminal search keeps highlighting the visible part of a match wrapped above the viewport", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.resize({.columns = 20, .rows = 6, .cellWidth = 8.0f, .cellHeight = 16.0f});
    session.text("clear; printf 'abcdefghijklmnopqrstuvwxyz0123\\n'; i=0; while [ $i -lt 30 ]; do echo line$i; "
                 "i=$((i+1)); done; sleep 30\r");
    REQUIRE(Shows(session, "line29"));
    session.search("abcdefghijklmnopqrstuvwxyz0123");
    const auto top = [&] {
        const auto& lines = session.screen().lines;
        return lines.empty() ? std::string() : RowText(lines.front());
    };
    for (const I32 direction : {1, -1}) {
        for (U64 step = 0; step < 60 && top().rfind("uvwxyz0123", 0) != 0; ++step) {
            session.scroll(direction, {10.0f, 10.0f}, {});
            session.poll();
        }
    }
    REQUIRE(top().rfind("uvwxyz0123", 0) == 0);
    const auto& row = session.screen().lines.front();
    const bool marked = row.match.has_value() || !row.matches.empty();
    REQUIRE(marked);
    const auto range = row.match.has_value() ? *row.match : row.matches.front();
    CHECK(range.start == 0);
    CHECK(range.end == 9);
}

TEST_CASE("Terminal sessions clear the alternate screen with the current pen instead of one an earlier app left", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(session.start({
        .program = "/bin/sh",
        .arguments = {"-c", "printf '\\033[?1049h\\033[48;2;0;0;0mstale\\033[?1049l'; "
                            "printf '\\033[?1049h\\033[0mready'; sleep 2"},
        .directory = "/",
    }) == Result::SUCCESS);
    REQUIRE(Shows(session, "ready"));
    U64 black = 0;
    for (const auto& row : session.screen().lines) {
        for (const auto& run : row.backgrounds) {
            black += run.color == ColorRGBA<F32>{0.0f, 0.0f, 0.0f, 1.0f} ? run.width : 0;
        }
    }
    CHECK(black == 0);
}

TEST_CASE("Terminal sessions publish the predominant background of the screen", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    CHECK(session.screen().fill == session.screen().background);
    CHECK(session.screen().base == session.screen().background);
    CHECK(session.screen().background.a == 1.0f);

    session.text("printf '\\033[44m\\033[2J\\033[Hblue'; sleep 30\r");
    REQUIRE(PollUntil(session, [&] { return session.screen().fill != session.screen().background; }));
    CHECK(session.screen().fill.b > session.screen().fill.r);
}

TEST_CASE("Terminal sessions follow default colors that apps set themselves", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    const auto base = session.screen().base;
    session.text("printf '\\033]10;rgb:00/ff/00\\007\\033]11;rgb:00/00/ff\\007\\033[2J\\033[Hcolored'; sleep 30\r");
    REQUIRE(PollUntil(session, [&] { return session.screen().background.b > 0.9f; }));
    CHECK(session.screen().base == base);
    CHECK(session.screen().fill == session.screen().background);
    REQUIRE(PollUntil(session, [&] { return RowText(session.screen().lines[0]).find("colored") == 0; }));
    bool green = false;
    for (const auto& span : session.screen().lines[0].spans) {
        green = green || (span.text.find("colored") != std::string::npos && span.color.g > 0.9f && span.color.r < 0.1f);
    }
    CHECK(green);
}

TEST_CASE("Terminal sessions count explicit edge cells in the default color toward the default background", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("printf '\\033[2J\\033[48;2;0;0;0m'; i=1; while [ $i -le 23 ]; do printf '\\033[%d;1H\\033[K' $i; "
                 "i=$((i+1)); done; printf '\\033[48;2;0;0;255m\\033[24;1H\\033[K\\033[0mfinished'; sleep 30\r");
    REQUIRE(PollUntil(session, [&] { return RowText(session.screen().lines.back()).rfind("finished", 0) == 0; }));
    U64 blue = 0;
    for (const auto& run : session.screen().lines.back().backgrounds) {
        blue += run.color.b > 0.9f ? run.width : 0;
    }
    REQUIRE(blue > 60);
    CHECK(session.screen().background == ColorRGBA<F32>{0.0f, 0.0f, 0.0f, 1.0f});
    CHECK(session.screen().fill == session.screen().background);
}

TEST_CASE("Terminal sessions ignore a large background block that does not touch the screen edges", kTags) {
    if (!TerminalSession::Supported()) {
        SKIP("The integrated terminal is not available in this build.");
    }
    TerminalSession session;
    REQUIRE(StartShell(session));
    session.text("printf '\\033[2J\\033[48;2;0;0;255m'; i=3; while [ $i -le 22 ]; do printf '\\033[%d;3H%76s' $i ''; "
                 "i=$((i+1)); done; printf '\\033[0m\\033[24;1Hfinished'; sleep 30\r");
    REQUIRE(PollUntil(session, [&] { return RowText(session.screen().lines.back()).rfind("finished", 0) == 0; }));
    U64 blue = 0;
    U64 edges = 0;
    const auto& lines = session.screen().lines;
    for (U64 row = 0; row < lines.size(); ++row) {
        for (const auto& run : lines[row].backgrounds) {
            if (run.color.b <= 0.9f) {
                continue;
            }
            blue += run.width;
            const bool edge = row == 0 || row + 1 == lines.size() || run.column == 0 ||
                              run.column + run.width >= session.screen().columns;
            edges += edge ? 1 : 0;
        }
    }
    REQUIRE(blue >= 20 * 76);
    REQUIRE(blue * 2 > static_cast<U64>(session.screen().columns) * session.screen().rows);
    CHECK(edges == 0);
    CHECK(session.screen().fill == session.screen().background);

    TerminalSession framed;
    REQUIRE(StartShell(framed));
    framed.text("printf '\\033[2J\\033[48;2;0;0;255m'; i=1; while [ $i -le 24 ]; do printf '\\033[%d;1H\\033[K' $i; "
               "i=$((i+1)); done; printf '\\033[0m'; sleep 30\r");
    REQUIRE(PollUntil(framed, [&] { return framed.screen().fill != framed.screen().background; }));
    CHECK(framed.screen().fill.b > 0.9f);
}
