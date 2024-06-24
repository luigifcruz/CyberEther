#ifndef JETSTREAM_BLOCK_REMOTE_BASE_HH
#define JETSTREAM_BLOCK_REMOTE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/remote.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Remote : public Block {
 public:
    // Configuration

    struct Config {
        std::string endpoint = "127.0.0.1:5000";
        Extent2D<U64> viewSize = {1280, 720};

        JST_SERDES(endpoint, viewSize);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        JST_SERDES();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "remote";
    }

    std::string name() const {
        return "Remote View";
    }

    std::string summary() const {
        return "Opens a remote view to a CyberEther instance.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Opens a remote view to a CyberEther instance via the network. This allows viewing and controlling the instance from a different computer.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            remote, "remote", {
                .endpoint = config.endpoint,
                .viewSize = config.viewSize,
            }, {},
            locale()
        ));

        // Initialize mouse state.

        lastMouseLeftDown = false;
        lastMouseRightDown = false;

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(remote->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawInfo() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Latency");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextFormatted("~{:.2f} ms", remote->statistics().latency);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Framebuffer");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextFormatted("{}x{} @ {:.0f} Hz", remote->getRemoteFramebufferSize().x, 
                                                  remote->getRemoteFramebufferSize().y,
                                                  remote->getRemoteFramerate());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Encoding");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextFormatted("{} (RGBA)", Viewport::VideoCodecToString(remote->getRemoteFramebufferCodec()));

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Status");
        ImGui::TableSetColumnIndex(1);
        ImGui::PushStyleColor(ImGuiCol_Text, remote->isBrokerConnected() ? ImVec4(0.196f, 0.804f, 0.194f, 1.0f) : 
                                                                           ImVec4(1.0f, 0.325f, 0.286f, 1.0f));
        ImGui::TextFormatted("{} {}", remote->isBrokerConnected() ? "Connected" : "Disconnected", 
                                      remote->isSocketStreaming() ? "Streaming" : "");
        ImGui::PopStyleColor();
    }

    constexpr bool shouldDrawInfo() const {
        return true;
    }

    void drawPreview(const F32& maxWidth) {
        const auto& size = remote->viewSize();
        const auto& ratio = size.ratio();
        const F32 width = (size.x < maxWidth) ? size.x : maxWidth;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((maxWidth - width) / 2.0f));
        ImGui::Image(remote->getTexture().raw(), ImVec2(width, width/ratio));
    }

    constexpr bool shouldDrawPreview() const {
        return remote->isBrokerConnected();
    }

    void drawView() {
        auto& io = ImGui::GetIO();
        const auto [x, y] = ImGui::GetContentRegionAvail();
        const auto scale = io.DisplayFramebufferScale;
        const auto [width, height] = remote->viewSize({
            static_cast<U64>(x*scale.x),
            static_cast<U64>(y*scale.y)
        });
        ImGui::Image(remote->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));
        const bool isImageHovered = ImGui::IsItemHovered();

        // Register mouse position.

        if (isImageHovered) {
            const auto mousePos = getRelativeMousePos();
            if (mousePos.x != lastMousePos.x || mousePos.y != lastMousePos.y) {
                remote->registerMousePos(mousePos.x*scale.x, mousePos.y*scale.y);
                lastMousePos = mousePos;
            }
        }

        // Register mouse buttons.

        if ((ImGui::IsMouseDown(ImGuiMouseButton_Left) && isImageHovered) != lastMouseLeftDown) {
            lastMouseLeftDown = !lastMouseLeftDown;
            remote->registerMouseButton(ImGuiMouseButton_Left, lastMouseLeftDown);
        }

        if ((ImGui::IsMouseDown(ImGuiMouseButton_Middle) && isImageHovered) != lastMouseMiddleDown) {
            lastMouseMiddleDown = !lastMouseMiddleDown;
            remote->registerMouseButton(ImGuiMouseButton_Middle, lastMouseMiddleDown);
        }

        if ((ImGui::IsMouseDown(ImGuiMouseButton_Right) && isImageHovered) != lastMouseRightDown) {
            lastMouseRightDown = !lastMouseRightDown;
            remote->registerMouseButton(ImGuiMouseButton_Right, lastMouseRightDown);
        }

        // Register mouse scroll.

        if (isImageHovered) {
            const F32 currentScrollY = io.MouseWheelH;
            const F32 currentScrollX = io.MouseWheel;
            if (currentScrollX != 0.0f || currentScrollY != 0.0f) {
                remote->registerMouseScroll(currentScrollX*scale.x, currentScrollY*scale.y);
            }
        }

        // Avoid moving the window when the mouse is over the view.

        if (isImageHovered && ImGui::IsAnyMouseDown()) {
            ImGui::GetCurrentWindow()->Flags |= ImGuiWindowFlags_NoMove;
        }

        // Register keyboard input.

        if (isImageHovered) {
            for (I32 i = ImGuiKey_NamedKey_BEGIN; i < ImGuiKey_NamedKey_END; i++) {
                const ImGuiKey key = static_cast<ImGuiKey>(i);
                if (ImGui::IsAliasKey(key)) {
                    continue;
                }
                auto& lastKey = lastKeyStates[i - ImGuiKey_NamedKey_BEGIN];
                if (ImGui::IsKeyDown(key) != lastKey) {
                    remote->registerKey(i, !lastKey);
                    lastKey = !lastKey;
                }
            }

            for (I32 i = 0; i < io.InputQueueCharacters.Size; i++) {
                const char key = io.InputQueueCharacters[i];
                if (key == '\0') {
                    break;
                }
                remote->registerChar(key);
            }
        }
    }

    constexpr bool shouldDrawView() const {
        return remote->isSocketStreaming();
    }

    constexpr bool shouldDrawFullscreen() const {
        return remote->isSocketStreaming();
    }

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Address");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##EndpointAddress", &config.endpoint, ImGuiInputTextFlags_EnterReturnsTrue)) {
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1);
        const bool isConnected = remote->isBrokerConnected();
        const F32 fullWidth = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button((isConnected) ? "Disconnected" : "Connect", ImVec2(fullWidth, 0))) {
            if (isConnected) {
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Disconnecting..." });
                    JST_CHECK_NOTIFY(remote->destroy());
                });
            } else {
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Connecting..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    bool lastKeyStates[ImGuiKey_NamedKey_COUNT] = { false };
    bool lastMouseLeftDown = false;
    bool lastMouseRightDown = false;
    bool lastMouseMiddleDown = false;
    ImVec2 lastMousePos = {0.0f, 0.0f};

    std::shared_ptr<Jetstream::Remote<D>> remote;

    ImVec2 getRelativeMousePos() {
        ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
        ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
        return ImVec2(mousePositionAbsolute.x - screenPositionAbsolute.x,
                      mousePositionAbsolute.y - screenPositionAbsolute.y);
    }

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Remote, is_specialized<Jetstream::Remote<D>>::value && 
                         std::is_same<IT, void>::value &&
                         std::is_same<OT, void>::value)

#endif
