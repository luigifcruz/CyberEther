#ifndef JETSTREAM_RENDER_MACROS_HH
#define JETSTREAM_RENDER_MACROS_HH

#include "jetstream/memory/macros.hh"
#include "jetstream/macros.hh"

#ifndef JST_CHECK_NOTIFY
#define JST_CHECK_NOTIFY(...) { \
    Result val = (__VA_ARGS__); \
    if (val == Result::ERROR) { \
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, JST_LOG_LAST_ERROR().c_str() }); \
    } else if (val == Result::FATAL) { \
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, JST_LOG_LAST_FATAL().c_str() }); \
    } else if (val == Result::WARNING) { \
        ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, JST_LOG_LAST_WARNING().c_str() }); \
    } else if (val == Result::SUCCESS) { \
        ImGui::InsertNotification({ ImGuiToastType_Success, 1000, "" }); \
    } \
}
#endif  // JST_CHECK_NOTIFY

#ifndef JST_MODULE_UPDATE
#define JST_MODULE_UPDATE(_MOD, ...) { \
    Result val = _MOD->__VA_ARGS__; \
    if (val == Result::RELOAD) { \
        JST_DISPATCH_ASYNC([&](){ \
            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading module..." }); \
            JST_CHECK_NOTIFY(instance().reloadModule(_MOD->locale())); \
        }); \
    } else if (val == Result::ERROR) { \
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, JST_LOG_LAST_ERROR().c_str() }); \
    } else if (val == Result::FATAL) { \
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, JST_LOG_LAST_FATAL().c_str() }); \
    } else if (val == Result::WARNING) { \
        ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, JST_LOG_LAST_WARNING().c_str() }); \
    } else if (val == Result::SUCCESS) { \
        ImGui::InsertNotification({ ImGuiToastType_Success, 1000, "" }); \
    } \
}
#endif  // JST_MODULE_UPDATE

#endif
