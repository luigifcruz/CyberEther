#include "jetstream/viewport/platforms/headless/remote.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

#include <memory>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <gst/webrtc/webrtc.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video-event.h>
#include <gst/gststructure.h>

#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>
#include <qrencode.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <string>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <vector>

namespace Jetstream::Viewport {

struct Remote::Impl {
    enum class Strategy {
        None,
        Software,
        HardwareNVENC,
        HardwareV4L2,
    };

    Config config;

    Device inputMemoryDevice = Device::None;
    Strategy encodingStrategy = Strategy::None;
    Device viewportDevice = Device::None;

    static std::string StrategyToString(const Strategy& strategy);

    // Broker

    std::string roomId;
    std::string consumerToken;
    std::string producerToken;
    std::vector<std::string> waitlist;
    std::vector<std::string> sessions;

    std::string brokerUrl;
    bool brokerRunning;
    std::thread brokerThread;
    std::unique_ptr<httplib::Client> brokerClient;

    Result createBroker();
    Result destroyBroker();

    Result createRoom();
    Result updateWaitlist();
    Result updateSessions();
    Result approveSession(const std::string& sessionId);

    void handleInput(const std::string& kind, const nlohmann::json& j);

    // Stream

    GstElement* pipeline;
    GstElement* source;
    GstElement* encoder;

    bool streaming = false;

    Result createStream();
    Result startStream();
    Result stopStream();
    Result destroyStream();

    Result checkGstreamerPlugins(const std::vector<std::string>& plugins,
                                 const bool& silent = false);
    static void onMessageCallback(GstWebRTCDataChannel* self, gchar* data, gpointer user_data);
    static void onChannelCallback(GstElement* self, GstWebRTCDataChannel* channel, gpointer user_data);
    static void rtcReadyCallback(GstElement* self, gchararray peer_id, GstElement* webrtcbin, gpointer udata);

    // Main Loop

    std::mutex bufferMutex;
    std::condition_variable bufferCond;
    bool bufferProcessed = false;

    bool forceKeyframe = false;
    std::chrono::time_point<std::chrono::steady_clock> initialFrameTime;
    std::chrono::time_point<std::chrono::steady_clock> lastKeyframeTime;

    static void OnBufferReleaseCallback(gpointer user_data);
};

Remote::Remote() {
    pimpl = std::make_unique<Impl>();
}

Remote::~Remote() {
    pimpl.reset();
}

Result Remote::create(const Viewport::Config& _config, const Device& _viewport_device) {
    JST_DEBUG("[REMOTE] Initializing plugin.");

    // Set variables.

    pimpl->config = _config;
    pimpl->viewportDevice = _viewport_device;

    // Validate configuration.

    if (pimpl->config.broker.empty()) {
        JST_ERROR("[REMOTE] Missing broker address.");
        return Result::ERROR;
    }

    // Create facilities.

    JST_CHECK(pimpl->createStream());
    JST_CHECK(pimpl->createBroker());

    return Result::SUCCESS;
}

Result Remote::destroy() {
    JST_DEBUG("[REMOTE] Destroying.");

    JST_CHECK(pimpl->destroyStream());
    JST_CHECK(pimpl->destroyBroker());

    return Result::SUCCESS;
}

class LogHistory final : public std::streambuf {
public:
    explicit LogHistory(std::size_t max_lines = 2000)
        : out_(this), max_lines_(max_lines) {
        setp(nullptr, nullptr);
    }

    std::ostream* sink() { return &out_; }

    std::vector<std::string> snapshot() const {
        std::lock_guard<std::mutex> lk(mu_);
        return std::vector<std::string>(lines_.begin(), lines_.end());
    }

    uint64_t version() const { return ver_.load(std::memory_order_relaxed); }

protected:
    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof()))
            return traits_type::not_eof(ch);
        const char c = traits_type::to_char_type(ch);
        appendChunk_(&c, 1);
        return traits_type::not_eof(ch);
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        appendChunk_(s, static_cast<std::size_t>(n));
        return n;
    }

    int sync() override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!pending_.empty()) {
            lines_.emplace_back(std::move(pending_));
            pending_.clear();
            trimUnlocked_();
            ver_.fetch_add(1, std::memory_order_relaxed);
        }
        return 0;
    }

private:
    void appendChunk_(const char* s, std::size_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        std::size_t i = 0;
        while (i < n) {
            const void* found = std::memchr(s + i, '\n', n - i);
            if (!found) {
                pending_.append(s + i, n - i);
                break;
            }
            const char* p = static_cast<const char*>(found);
            pending_.append(s + i, static_cast<std::size_t>(p - (s + i)));
            // Finish a line at '\n'
            lines_.emplace_back(std::move(pending_));
            pending_.clear();
            trimUnlocked_();
            ver_.fetch_add(1, std::memory_order_relaxed);
            i = static_cast<std::size_t>(p - s) + 1;
        }
    }

    void trimUnlocked_() {
        while (lines_.size() > max_lines_)
            lines_.pop_front();
    }

    mutable std::mutex mu_;
    std::deque<std::string> lines_;
    std::string pending_;
    std::size_t max_lines_;
    std::ostream out_;
    std::atomic<uint64_t> ver_{0};
};

Result Remote::Impl::createBroker() {
    // Connect to broker.

    {
        const auto protocol = (config.broker.find("localhost") != std::string::npos ||
                               config.broker.find("127.0.0.1") != std::string::npos) ? "http" : "https";
        brokerUrl = jst::fmt::format("{}://{}", protocol, config.broker);
        JST_INFO("[REMOTE] Connecting to broker at '{}'.", brokerUrl);
        brokerClient = std::make_unique<httplib::Client>(brokerUrl);
    }

    // Register room with broker.

    {
        // Check if broker is up.

        {
            auto res = brokerClient->Get("/health");
            if (!res || res->status != 200 || res->body != "ok") {
                JST_ERROR("[REMOTE] Failed to connect to server.");
                return Result::ERROR;
            }
            JST_DEBUG("[REMOTE] Connected to server.");
        }

        // Create room.

        JST_CHECK(createRoom());

        // Register credentials.

        brokerClient->set_bearer_token_auth(producerToken);
    }

    // Start socket.

    JST_CHECK(startStream());

    // Start broker interface.

    brokerRunning = true;
    brokerThread = std::thread([&]() {
        LogHistory log_history;
        JST_LOG_COLOR(false);
        JST_LOG_SET_SINK(log_history.sink());

        const std::string invite_url = jst::fmt::format("{}/remote#{}", brokerUrl, consumerToken);

        std::string clientAuthorizationCode;
        int logs_content_height = 0;

        auto CreateQRCode = [](const std::string& data) -> ftxui::Element {
            QRcode* qr = QRcode_encodeString8bit(data.c_str(), 0, QR_ECLEVEL_L);
            if (!qr) return ftxui::text("[QR encode error]");

            const int width = qr->width;
            const int border = 2;
            std::vector<ftxui::Element> lines;
            lines.reserve((width + 2 * border + 1) / 2);

            auto is_black = [&](int x, int y) -> bool {
                if (x < 0 || y < 0 || x >= width || y >= width) return false;
                return (qr->data[y * width + x] & 1) != 0;
            };

            for (int y = -border; y < width + border; y += 2) {
                std::string line;
                line.reserve(width + 2 * border);

                for (int x = -border; x < width + border; ++x) {
                    const bool upper = is_black(x, y);
                    const bool lower = is_black(x, y + 1);

                    if (upper && lower)      line += "█";
                    else if (upper)          line += "▀";
                    else if (lower)          line += "▄";
                    else                     line += " ";
                }
                lines.push_back(ftxui::text(std::move(line)));
            }

            QRcode_free(qr);
            return ftxui::vbox(std::move(lines)) | ftxui::center;
        };

        ftxui::InputOption input_options;
        input_options.on_change = [&] {
            std::string filtered;
            filtered.reserve(6);
            for (char c : clientAuthorizationCode) {
                if (filtered.size() < 6) {
                    filtered.push_back(std::toupper(static_cast<unsigned char>(c)));
                }
            }
            clientAuthorizationCode = std::move(filtered);
        };
        input_options.on_enter = [&]() {
            if (clientAuthorizationCode.size() != 6) {
                JST_ERROR("[REMOTE] Invalid Input: Code must be 6 digits.");
                clientAuthorizationCode.clear();
                return;
            }
            const auto code = clientAuthorizationCode;
            clientAuthorizationCode.clear();

            if (updateWaitlist() != Result::SUCCESS) {
                JST_ERROR("[REMOTE] Failed to update waitlist.");
                return;
            }

            auto to_lower = [](std::string_view s) {
                std::string out(s);
                std::transform(out.begin(), out.end(), out.begin(),
                               [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
                return out;
            };

            for (auto& sessionId : waitlist) {
                JST_DEBUG("[REMOTE] Candidate session: {}", sessionId);

                if (sessionId.ends_with(to_lower(code))) {
                    JST_INFO("[REMOTE] Client authorization code '{}' approved.", code);

                    if (approveSession(sessionId) != Result::SUCCESS) {
                        JST_ERROR("[REMOTE] Failed to approve session '{}'.", sessionId);
                        return;
                    }

                    if (updateSessions() != Result::SUCCESS) {
                        JST_ERROR("[REMOTE] Failed to update waitlist.");
                        return;
                    }

                    return;
                }
            }

            JST_ERROR("[REMOTE] Client authorization code '{}' not found.", code);
        };

        auto header = ftxui::Renderer([] {
            return ftxui::hbox({
                       ftxui::filler(),
                       ftxui::text(" CyberEther Remote ") | ftxui::bold,
                       ftxui::filler(),
                   }) | ftxui::bgcolor(ftxui::Color::White)
                      | ftxui::color(ftxui::Color::Black)
                      | ftxui::size(ftxui::HEIGHT, ftxui::EQUAL, 1);
        });

        auto left_panel = ftxui::Renderer([&] {
            auto url_display = ftxui::text(invite_url) | ftxui::underlined | ftxui::color(ftxui::Color::Blue);

            return ftxui::window(ftxui::text(" Join "), ftxui::vbox({
                ftxui::filler(),
                CreateQRCode(invite_url),
                ftxui::filler(),
                ftxui::separator(),
                ftxui::hbox(url_display) | ftxui::center,
                ftxui::text("Join with the QR code above or use the link!") | ftxui::dim | ftxui::center,
            })) | ftxui::xflex_grow | ftxui::yflex_grow;
        });

        auto logs_panel = ftxui::Renderer([&] {
            auto lines = log_history.snapshot();
            const int available_height = std::max(0, logs_content_height);
            const int line_count = std::min(available_height, static_cast<int>(lines.size()));
            const int start_index = static_cast<int>(lines.size()) - line_count;

            std::vector<ftxui::Element> rows;
            rows.reserve(available_height);

            for (int i = 0; i < available_height - line_count; ++i) {
                rows.push_back(ftxui::text(""));
            }

            for (int i = 0; i < line_count; ++i) {
                rows.push_back(ftxui::text(lines[start_index + i]));
            }

            return ftxui::window(ftxui::text(" Logs "), ftxui::vbox(std::move(rows))) | ftxui::yflex_grow;
        });

        auto input_field = ftxui::Input(&clientAuthorizationCode, "type the 6 digits code", input_options);

        auto room_panel = ftxui::Renderer([&] {
            return ftxui::window(
                    ftxui::text(" Room "),
                    ftxui::vbox({
                        ftxui::hbox(ftxui::text("Room ID: "),
                                    ftxui::text(roomId) | ftxui::bold),
#ifdef JST_DEBUG
                        ftxui::hbox(ftxui::text("Producer Token: "),
                                    ftxui::text(producerToken) | ftxui::bold),
                        ftxui::hbox(ftxui::text("Consumer Token: "),
                                    ftxui::text(consumerToken) | ftxui::bold),
#endif
                    }));
        });

        auto clients_panel = ftxui::Renderer([&] {
            std::vector<ftxui::Element> lines;
            if (sessions.empty()) {
                lines.push_back(ftxui::text("No clients yet.") | ftxui::dim | ftxui::bold);
            } else {
                for (const auto& s : sessions) {
                    lines.push_back(ftxui::text("• " + s));
                }
            }

            return ftxui::window(ftxui::text(" Clients "), ftxui::vbox(std::move(lines))) | ftxui::yflex_grow;
        });

        auto auth_container = ftxui::Container::Vertical({ input_field });

        auto auth_panel = ftxui::Renderer(auth_container, [&] {
            return ftxui::window(ftxui::text(" Add New Client "),
                                 input_field->Render() | ftxui::focusCursorBlockBlinking | ftxui::xflex_grow);
        });

        auto root2 = ftxui::Container::Vertical({
            auth_panel,
            room_panel,
            clients_panel,
        });

        auto session_panel = ftxui::Renderer(root2, [&] {
            return ftxui::vbox({
                auth_panel->Render(),
                room_panel->Render(),
                clients_panel->Render() | ftxui::yflex_grow,
            }) | ftxui::yflex_grow;
        });

        auto right_column = ftxui::Container::Vertical({logs_panel, session_panel});
        ftxui::Box right_box;

        auto right_panel = ftxui::Renderer(right_column, [&] {
            const int total_height = std::max(1, right_box.y_max - right_box.y_min + 1);
            const int logs_height = std::max(1, static_cast<int>(total_height * 0.65f));
            logs_content_height = std::max(0, logs_height - 2);

            return ftxui::vbox({
                logs_panel->Render() | ftxui::size(ftxui::HEIGHT, ftxui::EQUAL, logs_height),
                session_panel->Render() | ftxui::yflex_grow,
            }) | ftxui::xflex_grow | ftxui::yflex_grow | ftxui::reflect(right_box);
        });

        ftxui::Box body_box;
        auto body_container = ftxui::Container::Horizontal({left_panel, right_panel});
        auto body = ftxui::Renderer(body_container, [&] {
            const int total_width = std::max(1, body_box.x_max - body_box.x_min + 1);
            const int left_width  = static_cast<int>(std::round(total_width * 0.40f));
            const int right_width = std::max(0, total_width - left_width);

            return ftxui::hbox({
                       left_panel->Render()  | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, left_width),
                       right_panel->Render() | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, right_width),
                   }) | ftxui::xflex_grow | ftxui::yflex_grow | ftxui::reflect(body_box);
        });

        auto root = ftxui::Container::Vertical({header, body});
        auto app = ftxui::Renderer(root, [&] {
            return ftxui::vbox({
                header->Render(),
                body->Render() | ftxui::yflex_grow,
            }) | ftxui::xflex_grow | ftxui::yflex_grow;
        });

        auto screen = ftxui::ScreenInteractive::Fullscreen();

        auth_panel->SetActiveChild(input_field);
        input_field->TakeFocus();

        std::thread update_thread([&] {
            using namespace std::chrono_literals;
            uint64_t last_version = log_history.version();

            while (brokerRunning) {
                const auto current_version = log_history.version();
                if (current_version != last_version) {
                    last_version = current_version;
                    screen.Post(ftxui::Event::Custom);
                }
                std::this_thread::sleep_for(50ms);
            }
        });

        screen.Loop(app);

        brokerRunning = false;
        if (update_thread.joinable()) {
            update_thread.join();
        }

        JST_LOG_RESTORE_STDOUT();
        JST_LOG_COLOR(true);

        std::raise(SIGINT);

        return Result::SUCCESS;
    });

    return Result::SUCCESS;
}

Result Remote::Impl::destroyBroker() {
    JST_DEBUG("[REMOTE] Closing broker connection.");

    brokerRunning = false;
    if (brokerThread.joinable()) {
        brokerThread.join();
    }
    JST_CHECK(stopStream());

    return Result::SUCCESS;
}

Result Remote::Impl::createStream() {
    JST_DEBUG("[REMOTE] Creating stream.");

    // Initialize gstreamer.

    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    // Check if all plugins are available.

    std::vector<std::string> plugins = {
        "app",
        "rawparse",
        "coreelements",
    };

    // Inject socket endpoint plugins.

    plugins.push_back("rswebrtc");

    // Check required core plugins.

    JST_CHECK(checkGstreamerPlugins(plugins));

    // Inject codec plugins.

    std::vector<std::tuple<Device, Strategy, std::vector<std::string>>> combinations;

    if (config.codec == Viewport::VideoCodec::H264) {
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (viewportDevice == Device::Vulkan && Backend::State<Device::CUDA>()->isAvailable()) {
            combinations.push_back({Device::CUDA, Strategy::HardwareNVENC, {"nvcodec"}});
            GST_DEBUG("[REMOTE] Checking for NVENC strategy support for h264.");
        }
#endif

        if (checkGstreamerPlugins({"video4linux2"}, true) == Result::SUCCESS) {
            GstElementFactory* factory = gst_element_factory_find("v4l2h264enc");
            if (factory) {
                combinations.push_back({Device::CPU, Strategy::HardwareV4L2, {"video4linux2"}});
                gst_object_unref(GST_OBJECT(factory));
                GST_DEBUG("[REMOTE] Checking for V4L2 strategy support for h264.");
            }
        }

        combinations.push_back({Device::CPU, Strategy::Software, {"x264"}});
    }

    if (config.codec == Viewport::VideoCodec::VP8) {
        combinations.push_back({Device::CPU, Strategy::Software, {"vpx"}});
    }

    if (config.codec == Viewport::VideoCodec::VP9) {
        combinations.push_back({Device::CPU, Strategy::Software, {"vpx"}});
    }

    if (config.codec == Viewport::VideoCodec::AV1) {
        combinations.push_back({Device::CPU, Strategy::Software, {"rav1e"}});
    }

    for (const auto& [device, strategy, plugins] : combinations) {
        if ((strategy != Strategy::Software) && !config.hardwareAcceleration) {
            continue;
        }
        if (checkGstreamerPlugins(plugins, true) == Result::SUCCESS) {
            inputMemoryDevice = device;
            encodingStrategy = strategy;

            JST_INFO("[REMOTE] Using {} encoding with {} memory.", StrategyToString(strategy),
                                                                   GetDevicePrettyName(device));

            return Result::SUCCESS;
        }
        JST_DEBUG("[REMOTE] Failed to find plugins: {}", plugins);
    }

    JST_ERROR("[REMOTE] No encoding combination is available.");
    JST_ERROR("[REMOTE] This is tipically caused by missing plugins.");
    return Result::ERROR;
}

Result Remote::Impl::destroyStream() {
    JST_DEBUG("[REMOTE] Destroying stream.");

    encodingStrategy = Strategy::None;
    inputMemoryDevice = Device::None;

    return Result::SUCCESS;
}

Result Remote::Impl::checkGstreamerPlugins(const std::vector<std::string>& plugins,
                                       const bool& silent) {
    for (const auto& plugin : plugins) {
        if (!gst_registry_find_plugin(gst_registry_get(), plugin.c_str())) {
            if (!silent) {
                JST_ERROR("[REMOTE] Gstreamer plugin '{}' is not available.", plugin);
            }
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

void Remote::Impl::handleInput(const std::string& kind, const nlohmann::json& j) {
    // Helper: Convert DOM mouse button to ImGui button index.
    auto map_mouse_button = [](int dom_button) -> int {
        // DOM: 0=Left, 1=Middle, 2=Right; ImGui: 0=Left, 1=Right, 2=Middle
        switch (dom_button) {
            case 0: return 0;
            case 1: return 2;
            case 2: return 1;
            default: return dom_button;  // Extra buttons (rare)
        }
    };

    // Helper: Denormalize [0..1] -> pixels.
    auto denorm = [](double n, int extent) -> float {
        if (extent <= 0) return 0.0f;
        if (n >= 0.0 && n <= 1.0) {
            return static_cast<float>(std::round(n * std::max(0, extent - 1)));
        }
        return static_cast<float>(std::round(n));
    };

    // Helper: Quick first-codepoint (ASCII/UTF-8) -> Unicode scalar for ImGui::AddInputCharacter.
    auto first_codepoint = [](const std::string& s) -> unsigned int {
        if (s.empty()) return 0u;
        if (s.size() > 1) return 0u;

        const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
        // ASCII fast path
        if (p[0] < 0x80) return static_cast<unsigned int>(p[0]);
        // Minimal UTF-8 decode for leading bytes 110xxxxx, 1110xxxx, 11110xxx
        if ((p[0] & 0xE0) == 0xC0 && s.size() >= 2) {
            return ((p[0] & 0x1F) << 6) | (p[1] & 0x3F);
        }
        if ((p[0] & 0xF0) == 0xE0 && s.size() >= 3) {
            return ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        }
        if ((p[0] & 0xF8) == 0xF0 && s.size() >= 4) {
            return ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        }
        return 0u;
    };

    // Helper: map JS KeyboardEvent.{code,key} → ImGuiKey.
    auto map_key = [](const std::string& code, const std::string& key) -> ImGuiKey {
        // Letter keys: "KeyA".."KeyZ"
        if (code.size() == 4 && code.rfind("Key", 0) == 0) {
            char c = static_cast<char>(std::toupper(static_cast<unsigned char>(code[3])));
            if (c >= 'A' && c <= 'Z') return static_cast<ImGuiKey>(ImGuiKey_A + (c - 'A'));
        }
        // Digit keys: "Digit0".."Digit9"
        if (code.rfind("Digit", 0) == 0 && code.size() == 6 && std::isdigit(static_cast<unsigned char>(code[5]))) {
            char d = code[5];
            return static_cast<ImGuiKey>(ImGuiKey_0 + (d - '0'));
        }
        // Function keys: "F1".."F24"
        if (!code.empty() && code[0] == 'F' && code.size() <= 3 && std::isdigit(static_cast<unsigned char>(code[1]))) {
            int f = std::clamp(std::stoi(code.substr(1)), 1, 24);
            return static_cast<ImGuiKey>(ImGuiKey_F1 + (f - 1));
        }
        // Arrows
        if (code == "ArrowLeft")  return ImGuiKey_LeftArrow;
        if (code == "ArrowRight") return ImGuiKey_RightArrow;
        if (code == "ArrowUp")    return ImGuiKey_UpArrow;
        if (code == "ArrowDown")  return ImGuiKey_DownArrow;

        // Editing / navigation
        if (code == "Enter" || key == "Enter")          return ImGuiKey_Enter;
        if (code == "Escape" || key == "Escape")        return ImGuiKey_Escape;
        if (code == "Backspace" || key == "Backspace")  return ImGuiKey_Backspace;
        if (code == "Tab" || key == "Tab")              return ImGuiKey_Tab;
        if (code == "Space" || key == " ")              return ImGuiKey_Space;
        if (code == "Delete")                           return ImGuiKey_Delete;
        if (code == "Insert")                           return ImGuiKey_Insert;
        if (code == "Home")                             return ImGuiKey_Home;
        if (code == "End")                              return ImGuiKey_End;
        if (code == "PageUp")                           return ImGuiKey_PageUp;
        if (code == "PageDown")                         return ImGuiKey_PageDown;

        // Punctuation / symbols (best-effort)
        if (code == "Minus")        return ImGuiKey_Minus;
        if (code == "Equal")        return ImGuiKey_Equal;
        if (code == "BracketLeft")  return ImGuiKey_LeftBracket;
        if (code == "BracketRight") return ImGuiKey_RightBracket;
        if (code == "Backslash")    return ImGuiKey_Backslash;
        if (code == "Semicolon")    return ImGuiKey_Semicolon;
        if (code == "Quote")        return ImGuiKey_Apostrophe;
        if (code == "Comma")        return ImGuiKey_Comma;
        if (code == "Period")       return ImGuiKey_Period;
        if (code == "Slash")        return ImGuiKey_Slash;
        if (code == "Backquote" || code == "Backtick" || code == "QuoteLeft") return ImGuiKey_GraveAccent;

        return ImGuiKey_None;
    };

    //
    // Mouse Handler
    //

    if (kind == "mouse") {
        const std::string act = j.value("act", "");
        const double nx = j.value("x", 0.0);
        const double ny = j.value("y", 0.0);

        const int w = config.size.x;
        const int h = config.size.y;
        const float px = denorm(nx, w);
        const float py = denorm(ny, h);

        ImGuiIO& io = ImGui::GetIO();

        if (act == "move") {
            JST_TRACE("[REMOTE] Mouse: move (x='{}', y='{}', nx='{}', ny='{}')", px, py, nx, ny);
            io.AddMousePosEvent(px, py);
            return;
        }

        if (act == "down" || act == "up" || act == "click" || act == "dblclick") {
            const int dom_button = j.value("button", 0);
            const int b = map_mouse_button(dom_button);

            // Ensure position is up-to-date at click time.
            io.AddMousePosEvent(px, py);

            if (act == "down") {
                JST_TRACE("[REMOTE] Mouse: down (b='{}', x='{}', y='{}')", b, px, py);
                io.AddMouseButtonEvent(b, true);
                return;
            }
            if (act == "up") {
                JST_TRACE("[REMOTE] Mouse: up b={}, x={}, y={}", b, px, py);
                io.AddMouseButtonEvent(b, false);
                return;
            }
            if (act == "click" || act == "dblclick") {
                JST_TRACE("[REMOTE] Mouse: ignore (b='{}', x='{}', y='{}')", b, px, py);
                return;
            }
        }

        JST_TRACE("[REMOTE] Mouse: unknown (act='{}')", act);
        return;
    }

    //
    // Mouse Wheel Handler
    //

    if (kind == "wheel") {
        const int   deltaMode = j.value("deltaMode", 0);  // 0=pixel, 1=line, 2=page
        double      dx = j.value("deltaX", 0.0);
        double      dy = j.value("deltaY", 0.0);

        // Scale to "steps".
        if (deltaMode == 0) { dx /= 100.0; dy /= 100.0; }
        else if (deltaMode == 2) { dx *= 3.0; dy *= 3.0; }

        // Invert vertical for ImGui convention.
        dy = -dy;

        ImGuiIO& io = ImGui::GetIO();
        io.AddMouseWheelEvent(static_cast<float>(dx), static_cast<float>(dy));
        JST_TRACE("[REMOTE] Wheel: (dx='{}', dy='{}', mode='{}')", dx, dy, deltaMode);
        return;
    }

    //
    // Keyboard Handler
    //

    const bool alt   = j.value("altKey",   false);
    const bool ctrl  = j.value("ctrlKey",  false);
    const bool shift = j.value("shiftKey", false);
    const bool meta  = j.value("metaKey",  false);

    if (kind == "keyboard") {
        // Modifiers.

        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiKey_LeftAlt, alt);
        io.AddKeyEvent(ImGuiKey_LeftCtrl, ctrl);
        io.AddKeyEvent(ImGuiKey_LeftShift, shift);
        io.AddKeyEvent(ImGuiKey_LeftSuper, meta);

        const std::string action = j.value("action", "");  // "down" | "up"
        const std::string code = j.value("code", "");      // e.g. "KeyA", "Digit1", "ArrowLeft"
        const std::string key = j.value("key", "");        // printable, or names like "Enter"
        const bool pressed = (action == "down");

        const ImGuiKey k = map_key(code, key);
        if (k != ImGuiKey_None) {
            ImGui::GetIO().AddKeyEvent(k, pressed);
        }
        JST_TRACE("[REMOTE] Keyboard: event (pressed='{}', key='{}', code='{}', ImGuiKey='{}')", pressed ? "down" : "up", code, key, (int)k);

        // Text input: Add on key-down when it looks printable and no conflicting modifiers.
        if (pressed && !ctrl && !alt && !meta) {
            const unsigned int cp = first_codepoint(key);
            if (cp >= 0x20 && cp != 0x7F) {
                ImGui::GetIO().AddInputCharacter(cp);
                JST_TRACE("[REMOTE] Keyboard: char (key='{}', code='U+{:04X}')", key, cp);
            }
        }
        return;
    }

    JST_TRACE("[REMOTE] Unknown control (kind='{}').", kind);
}

void Remote::Impl::onMessageCallback(GstWebRTCDataChannel* self, gchar* data, gpointer user_data) {
    (void)self;

    JST_TRACE("[REMOTE] - Received string: {}", data);

    nlohmann::json j;

    try {
        j = nlohmann::json::parse(data);
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Bad control JSON: {} (payload='{}')", e.what(), data);
        return;
    }

    const std::string kind = j.value("kind", "");
    if (kind.empty()) {
        JST_WARN("[REMOTE] Control msg without 'kind': '{}'", data);
        return;
    }

    auto* that = reinterpret_cast<Remote::Impl*>(user_data);
    that->handleInput(kind, j);
}

void Remote::Impl::onChannelCallback(GstElement* self, GstWebRTCDataChannel* channel, gpointer user_data) {
    (void)self;
    (void)user_data;

    gchar *label = NULL; gint id = -1; gboolean negotiated = FALSE;
    g_object_get(channel, "label", &label, "id", &id, "negotiated", &negotiated, NULL);
    JST_INFO("[REMOTE] Dropping data channel opened from client (label='{}', id='{}', negotiated='{}')", label, id, negotiated);

    g_signal_emit_by_name(channel, "close");
    g_object_unref(channel);
    g_free(channel);
}

void Remote::Impl::rtcReadyCallback(GstElement* self, gchararray peer_id, GstElement* webrtcbin, gpointer udata) {
    (void)self;

    JST_INFO("[REMOTE] New consumer ({}) connected.", peer_id);

    // Install consumer created data channels callback.

    g_signal_connect(G_OBJECT(webrtcbin), "on-data-channel", G_CALLBACK(onChannelCallback), udata);

    // Create data channel for control messages.

    GstWebRTCDataChannel* channel = NULL;
    g_signal_emit_by_name(G_OBJECT(webrtcbin), "create-data-channel", "control", nullptr, &channel);
    if (!channel) {
        JST_ERROR("[REMOTE] Failed to create data channel.");
        return;
    }
    g_signal_connect(G_OBJECT(channel), "on-message-string", G_CALLBACK(onMessageCallback), udata);
}

Result Remote::Impl::startStream() {
    JST_DEBUG("[REMOTE] Starting stream.");

    // Create pipeline.

    pipeline = gst_pipeline_new("headless-src-pipeline");

    if (!pipeline) {
        JST_ERROR("[REMOTE] Failed to create gstreamer pipeline.");
        return Result::ERROR;
    }

    // Create elements.

    std::map<std::string, GstElement*> elements;
    std::vector<std::string> elementOrder;

    elements["source"] = source = gst_element_factory_make("appsrc", "source");
    elementOrder.push_back("source");

    elements["caps"] = gst_element_factory_make("capsfilter", "caps");
    elementOrder.push_back("caps");

    if (encodingStrategy == Strategy::Software || encodingStrategy == Strategy::HardwareV4L2) {
        elements["rawparser"] = gst_element_factory_make("rawvideoparse", "rawparser");
        elementOrder.push_back("rawparser");

        elements["convert"] = gst_element_factory_make("videoconvert", "convert");
        elementOrder.push_back("convert");
    }

    if (config.codec == Viewport::VideoCodec::H264) {
        if (encodingStrategy == Strategy::HardwareNVENC) {
            elements["encoder"] = encoder = gst_element_factory_make("nvh264enc", "encoder");
            elementOrder.push_back("encoder");
        }

        if (encodingStrategy == Strategy::HardwareV4L2) {
            elements["encoder"] = encoder = gst_element_factory_make("v4l2h264enc", "encoder");
            elementOrder.push_back("encoder");
        }

        if (encodingStrategy == Strategy::Software) {
            elements["encoder"] = encoder = gst_element_factory_make("x264enc", "encoder");
            elementOrder.push_back("encoder");
        }

        elements["hwcaps"] = gst_element_factory_make("capsfilter", "hwcaps");
        elementOrder.push_back("hwcaps");

        elements["parser"] = gst_element_factory_make("h264parse", "parser");
        elementOrder.push_back("parser");
    }

    if (config.codec == Viewport::VideoCodec::VP8) {
        elements["encoder"] = encoder = gst_element_factory_make("vp8enc", "encoder");
        elementOrder.push_back("encoder");
    }

    if (config.codec == Viewport::VideoCodec::VP9) {
        elements["encoder"] = encoder = gst_element_factory_make("vp9enc", "encoder");
        elementOrder.push_back("encoder");
    }

    if (config.codec == Viewport::VideoCodec::AV1) {
        elements["encoder"] = encoder = gst_element_factory_make("rav1enc", "encoder");
        elementOrder.push_back("encoder");
    }

    elements["webrtc"] = gst_element_factory_make("webrtcsink", "webrtc");
    elementOrder.push_back("webrtc");

    for (const auto& [name, element] : elements) {
        if (!element) {
            JST_ERROR("[REMOTE] Failed to create gstreamer element '{}'.", name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }
    }

    // Configure elements.

    g_object_set(elements["source"], "block", true, nullptr);
    g_object_set(elements["source"], "format", 3, nullptr);
    g_object_set(elements["source"], "leaky-type", 2, nullptr);
    g_object_set(elements["source"], "is-live", true, nullptr);
    g_object_set(elements["source"], "max-bytes", 2*config.size.x*config.size.y*4, nullptr);

    GstCaps* caps = gst_caps_new_simple("video/x-raw",
                                        "format", G_TYPE_STRING, "BGRA",
                                        "width", G_TYPE_INT, config.size.x,
                                        "height", G_TYPE_INT, config.size.y,
                                        "framerate", GST_TYPE_FRACTION, config.framerate, 1,
                                        "interlace-mode", G_TYPE_STRING, "progressive",
                                        "colorimetry", G_TYPE_STRING, "bt709",
                                        nullptr);

    if (encodingStrategy == Strategy::HardwareNVENC && inputMemoryDevice == Device::CUDA) {
        GstCapsFeatures *features = gst_caps_features_new("memory:CUDAMemory", nullptr);
        gst_caps_set_features(caps, 0, features);
    }

    g_object_set(elements["caps"], "caps", caps, nullptr);
    gst_caps_unref(caps);

    if (encodingStrategy == Strategy::Software || encodingStrategy == Strategy::HardwareV4L2) {
        g_object_set(elements["rawparser"], "use-sink-caps", 0, nullptr);
        g_object_set(elements["rawparser"], "format", 12, nullptr);
        g_object_set(elements["rawparser"], "width", config.size.x, nullptr);
        g_object_set(elements["rawparser"], "height", config.size.y, nullptr);
        // TODO: Unclear if this is needed. Crash on some systems.
        //g_object_set(elements["rawparser"], "framerate", 1.0f/config.framerate, nullptr);
    }

    if (config.codec == Viewport::VideoCodec::H264) {
        if (encodingStrategy == Strategy::HardwareNVENC) {
            g_object_set(elements["encoder"], "zerolatency", true, nullptr);
            g_object_set(elements["encoder"], "preset", 5, nullptr);
        }

        if (encodingStrategy == Strategy::HardwareV4L2) {
        }

        if (encodingStrategy == Strategy::Software) {
            g_object_set(elements["encoder"], "speed-preset", 1, nullptr);
            g_object_set(elements["encoder"], "tune", 4, nullptr);
            g_object_set(elements["encoder"], "bitrate", 25*1024, nullptr);
        }

        GstCaps* hwcaps = gst_caps_new_simple("video/x-h264",
                                              "level", G_TYPE_STRING, "5",
                                              "profile", G_TYPE_STRING, "baseline",
                                              nullptr);
        g_object_set(elements["hwcaps"], "caps", hwcaps, nullptr);
        gst_caps_unref(hwcaps);

        g_object_set(elements["parser"], "config-interval", 1, nullptr);
    }

    if (config.codec == Viewport::VideoCodec::AV1) {
        g_object_set(elements["encoder"], "low-latency", true, nullptr);
        g_object_set(elements["encoder"], "speed-preset", 10, nullptr);
        g_object_set(elements["encoder"], "bitrate", 25*1024*1024, nullptr);
    }

    if (config.codec == Viewport::VideoCodec::VP8 ||
        config.codec == Viewport::VideoCodec::VP9) {
        g_object_set(elements["encoder"], "target-bitrate", 25*1024*1024, nullptr);
    }

    g_object_set(elements["webrtc"], "signalling-server-host", "localhost", nullptr);

    GstStructure *s = gst_structure_new_empty("meta");
    gst_structure_set(s, "token", G_TYPE_STRING, producerToken.c_str(), nullptr);
    g_object_set(elements["webrtc"], "meta", s, NULL);
    gst_structure_free(s);

    GObject* signaller;
    g_object_get(elements["webrtc"], "signaller", &signaller, nullptr);
    if (signaller) {
        g_signal_connect(G_OBJECT(signaller), "webrtcbin-ready", G_CALLBACK(rtcReadyCallback), this);
        g_object_unref(signaller);
    } else {
        JST_ERROR("[REMOTE] Failed to get signaller object from WebRTC element.");
    }

    // Add elements to pipeline.

    for (const auto& [name, element] : elements) {
        if (!gst_bin_add(GST_BIN(pipeline), element)) {
            JST_ERROR("[REMOTE] Failed to add gstreamer element '{}' to pipeline.", name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }
    }

    // Link elements.

    std::string lastElement = "source";
    for (const auto& name : elementOrder) {
        if (name == "source") {
            continue;
        }

        if (!gst_element_link(elements[lastElement], elements[name])) {
            JST_ERROR("[REMOTE] Failed to link gstreamer element '{}' -> '{}'.", lastElement, name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }

        lastElement = name;
    }

    // Start pipeline.

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        JST_ERROR("[REMOTE] Failed to start gstreamer pipeline.");
        gst_object_unref(pipeline);
        return Result::ERROR;
    }

    initialFrameTime = std::chrono::steady_clock::now();
    forceKeyframe = true;
    streaming = true;

    return Result::SUCCESS;
}

Result Remote::Impl::stopStream() {
    JST_DEBUG("[REMOTE] Stopping stream.");

    // Stop pipeline.

    if (streaming) {
        // Stop piping frames.
        streaming = false;

        // Send EOS.
        gst_element_send_event(pipeline, gst_event_new_eos());

        // Wait pipeline to process EOS.

        GstBus* bus = gst_element_get_bus(pipeline);
        GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_EOS);
        if (msg) {
            gst_message_unref(msg);
        }
        gst_object_unref(bus);

        // Stop pipeline.
        gst_element_set_state(pipeline, GST_STATE_NULL);

        // Cleanup.
        gst_object_unref(pipeline);
    }

    return Result::SUCCESS;
}

//
// Main Loop
//

void Remote::Impl::OnBufferReleaseCallback(gpointer user_data) {
    auto* that = reinterpret_cast<Remote::Impl*>(user_data);
    std::unique_lock<std::mutex> lock(that->bufferMutex);
    that->bufferProcessed = true;
    that->bufferCond.notify_one();
}

Result Remote::pushNewFrame(const void* data) {
    if (!pimpl->streaming) {
        return Result::SUCCESS;
    }

    // Create buffer.

    GstBuffer* buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                                    const_cast<void*>(data),
                                                    pimpl->config.size.x * pimpl->config.size.y * 4,
                                                    0,
                                                    pimpl->config.size.x * pimpl->config.size.y * 4,
                                                    pimpl.get(),
                                                    &Impl::OnBufferReleaseCallback);

    // Calculate timings.

    const auto currentFrameTime = std::chrono::steady_clock::now();
    const auto elapsedSinceLastFrame = std::chrono::duration_cast<std::chrono::nanoseconds>(currentFrameTime -
                                                                                            pimpl->initialFrameTime);
    const auto elapsedSinceLastKeyframe = std::chrono::duration_cast<std::chrono::seconds>(currentFrameTime -
                                                                                            pimpl->lastKeyframeTime);

    // Set buffer timings (PTS and DTS).

    GST_BUFFER_PTS(buffer) = static_cast<U64>(elapsedSinceLastFrame.count());
    GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;

    // Force keyframe every 1 seconds.

    if ((elapsedSinceLastKeyframe.count() > 1) || pimpl->forceKeyframe) {
        GstEvent* force_key_unit_event = gst_video_event_new_downstream_force_key_unit(
            GST_CLOCK_TIME_NONE,
            GST_CLOCK_TIME_NONE,
            GST_CLOCK_TIME_NONE,
            TRUE,
            0
        );

        gst_element_send_event(pimpl->encoder, force_key_unit_event);

        pimpl->lastKeyframeTime = currentFrameTime;
        pimpl->forceKeyframe = false;
    }

    // Push frame to pipeline.

    if (gst_app_src_push_buffer(GST_APP_SRC(pimpl->source), buffer) != GST_FLOW_OK) {
        JST_ERROR("[REMOTE] Failed to push buffer to gstreamer pipeline.");
        return Result::ERROR;
    }

    // Wait for buffer to be processed.

    {
        std::unique_lock<std::mutex> lock(pimpl->bufferMutex);
        pimpl->bufferCond.wait(lock, [&]{ return pimpl->bufferProcessed; });
        pimpl->bufferProcessed = false;
    }

    return Result::SUCCESS;
}

//
// API
//

Result Remote::Impl::createRoom() {
    auto params = httplib::Params{};
    auto res = brokerClient->Post("/v1/room/create", params);
    if (!res || res->status != 201) {
        JST_ERROR("[REMOTE] Failed to create room.");
        return Result::ERROR;
    }

    try {
        auto j = nlohmann::json::parse(res->body);

        if (!j.contains("roomId")) {
            JST_ERROR("[REMOTE] Missing field 'roomId': {}", res->body);
            return Result::ERROR;
        }

        if (!j.contains("producerToken")) {
            JST_ERROR("[REMOTE] Missing field 'producerToken': {}", res->body);
            return Result::ERROR;
        }

        if (!j.contains("consumerToken")) {
            JST_ERROR("[REMOTE] Missing field 'consumerToken': {}", res->body);
            return Result::ERROR;
        }

        roomId = j["roomId"].get<std::string>();
        producerToken = j["producerToken"].get<std::string>();
        consumerToken = j["consumerToken"].get<std::string>();
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] JSON parse error '{}': {}", e.what(), res->body);
        return Result::ERROR;
    }

    JST_DEBUG("[REMOTE] New room created.");
    return Result::SUCCESS;
}

Result Remote::Impl::updateWaitlist() {
    auto res = brokerClient->Get("/v1/room/waitlist");
    if (!res || res->status != 200) {
        JST_ERROR("[REMOTE] Failed to pull waitlist.");
        return Result::ERROR;
    }

    try {
        auto j = nlohmann::json::parse(res->body);

        if (!j.contains("sessions")) {
            JST_ERROR("[REMOTE] Missing field 'sessions': {}", res->body);
            return Result::ERROR;
        }

        waitlist = j["sessions"].get<std::vector<std::string>>();
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] JSON parse error '{}': {}", e.what(), res->body);
        return Result::ERROR;
    }

    JST_DEBUG("[REMOTE] Waitlist updated.");
    return Result::SUCCESS;
}

Result Remote::Impl::updateSessions() {
    auto res = brokerClient->Get("/v1/room/sessions");
    if (!res || res->status != 200) {
        JST_ERROR("[REMOTE] Failed to pull sessions.");
        return Result::ERROR;
    }

    try {
        auto j = nlohmann::json::parse(res->body);

        if (!j.contains("sessions")) {
            JST_ERROR("[REMOTE] Missing field 'sessions': {}", res->body);
            return Result::ERROR;
        }

        sessions = j["sessions"].get<std::vector<std::string>>();
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] JSON parse error '{}': {}", e.what(), res->body);
        return Result::ERROR;
    }

    JST_DEBUG("[REMOTE] Sessions updated.");
    return Result::SUCCESS;
}

Result Remote::Impl::approveSession(const std::string& sessionId) {
    auto json = nlohmann::json{
        {"sessionId", sessionId}
    };
    auto res = brokerClient->Post("/v1/room/approve", json.dump(), "application/json");
    if (!res || res->status != 200) {
        JST_ERROR("[REMOTE] Failed to post approve.");
        return Result::ERROR;
    }

    JST_DEBUG("[REMOTE] Session approved.");
    return Result::SUCCESS;
}

//
// Helpers
//

const Device& Remote::inputMemoryDevice() const {
    return pimpl->inputMemoryDevice;
}

std::string Remote::Impl::StrategyToString(const Strategy& strategy) {
    switch (strategy) {
        case Strategy::None:
            return "None";
        case Strategy::Software:
            return "Software";
        case Strategy::HardwareNVENC:
            return "Hardware NVIDIA (NVENC)";
        case Strategy::HardwareV4L2:
            return "Hardware Linux (V4L2)";
        default:
            return "Unknown";
    }
}

}  // namespace Jetstream::Viewport
