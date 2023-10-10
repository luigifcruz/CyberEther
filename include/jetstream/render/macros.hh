#ifndef JETSTREAM_RENDER_MACROS_HH
#define JETSTREAM_RENDER_MACROS_HH

#include "jetstream/memory/macros.hh"

#ifndef JST_CHECK_NOTIFY
#define JST_CHECK_NOTIFY(...) { \
    Result val = (__VA_ARGS__); \
    if (val == Result::ERROR) { \
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, JST_LOG_LAST_ERROR().c_str() }); \
    } else if (val == Result::FATAL) { \
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, JST_LOG_LAST_FATAL().c_str() }); \
    } else if (val == Result::SUCCESS) { \
        ImGui::InsertNotification({ ImGuiToastType_Success, 1000, "" }); \
    } \
}
#endif  // JST_CHECK_NOTIFY

#endif
