#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH

#include "jetstream/config.hh"
#include "jetstream/render/sakura/components/chat_composer.hh"
#include "jetstream/render/sakura/typography.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/surface.hh"

#include <array>
#include <random>

namespace Jetstream {

struct AssistantComposer {
    static constexpr KeyCode VoiceKey = KeyCode::J;
#if defined(JST_OS_MAC)
    static constexpr const char* VoiceHint = "Start voice (Cmd+J)";
#else
    static constexpr const char* VoiceHint = "Start voice (Ctrl+J)";
#endif

    static constexpr std::array<const char*, 12> Prompts = {
        "Build an FM radio receiver.",
        "Show me the signals around me.",
        "Track the aircraft flying overhead on a globe.",
        "Build a spectrum analyzer for my SDR.",
        "Generate a QPSK signal and show its constellation.",
        "Show me how noise affects a clean tone.",
        "Teach me how a spectrum analyzer works.",
        "Build a waterfall of the Wi-Fi band.",
        "Listen to pilots on the aircraft band.",
        "Show me the difference between AM and FM.",
        "Show me what clipping does to a sine wave.",
        "Stream my SDR to another computer.",
    };

    static const char* RandomPrompt();
    static Sakura::ChatComposer::Style Style(Sakura::ChatComposer::Style style,
                                             const char* placeholder = "Ask anything...");
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH
