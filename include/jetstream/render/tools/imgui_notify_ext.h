// imgui-notify by patrickcjk
// https://github.com/patrickcjk/imgui-notify
// MIT License
// Modified by Luigi.

#ifndef IMGUI_NOTIFY
#define IMGUI_NOTIFY

#include "jetstream/render/tools/imgui.h"

#include <vector>
#include <string>
#include <chrono>
#include <cstdarg>
#include <cstdio>

#define NOTIFY_MAX_MSG_LENGTH           (uint64_t)4096   // Max message content length
#define NOTIFY_PADDING_X                (float)20.0f     // Bottom-left X padding
#define NOTIFY_PADDING_Y                (float)20.0f     // Bottom-left Y padding
#define NOTIFY_PADDING_MESSAGE_Y        (float)10.0f     // Padding Y between each message
#define NOTIFY_FADE_IN_OUT_TIME         (uint64_t)150    // Fade in and out duration
#define NOTIFY_DEFAULT_DISMISS          (uint64_t)3000   // Auto dismiss after X ms (default, applied only of no data provided in constructors)
#define NOTIFY_OPACITY                  (float)1.0f      // 0-1 Toast opacity
#define NOTIFY_TOAST_FLAGS              ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoSavedSettings
#define NOTIFY_NULL_OR_EMPTY(str)       (!str ||! strlen(str))
#define NOTIFY_FORMAT(fn, format, ...)	if (format) { va_list args; va_start(args, format); fn(format, args); va_end(args); }

typedef int ImGuiToastType;
typedef int ImGuiToastPhase;
typedef int ImGuiToastPos;

enum ImGuiToastType_
{
    ImGuiToastType_None,
    ImGuiToastType_Success,
    ImGuiToastType_Warning,
    ImGuiToastType_Error,
    ImGuiToastType_Info,
    ImGuiToastType_COUNT
};

enum ImGuiToastPhase_
{
    ImGuiToastPhase_FadeIn,
    ImGuiToastPhase_Wait,
    ImGuiToastPhase_FadeOut,
    ImGuiToastPhase_Expired,
    ImGuiToastPhase_COUNT
};

enum ImGuiToastPos_
{
    ImGuiToastPos_TopLeft,
    ImGuiToastPos_TopCenter,
    ImGuiToastPos_TopRight,
    ImGuiToastPos_BottomLeft,
    ImGuiToastPos_BottomCenter,
    ImGuiToastPos_BottomRight,
    ImGuiToastPos_Center,
    ImGuiToastPos_COUNT
};

class ImGuiToast
{
private:
    ImGuiToastType  type = ImGuiToastType_None;
    char            title[NOTIFY_MAX_MSG_LENGTH];
    char            content[NOTIFY_MAX_MSG_LENGTH];
    int             dismiss_time = NOTIFY_DEFAULT_DISMISS;
    uint64_t        creation_time = 0;

private:
    // Setters

    inline auto set_title(const char* format, va_list args) { vsnprintf(this->title, sizeof(this->title), format, args); }

    inline auto set_content(const char* format, va_list args) { vsnprintf(this->content, sizeof(this->content), format, args); }

public:

    inline void set_title(const char* format, ...) { NOTIFY_FORMAT(this->set_title, format); }

    inline void set_content(const char* format, ...) { NOTIFY_FORMAT(this->set_content, format); }

    inline void set_type(const ImGuiToastType& type) { IM_ASSERT(type < ImGuiToastType_COUNT); this->type = type; };

public:
    // Getters

    inline const char* get_title() { return this->title; };

    const char* get_default_title() {
        if (!strlen(this->title))
        {
            switch (this->type)
            {
            case ImGuiToastType_None:
                return NULL;
            case ImGuiToastType_Success:
                return "Success";
            case ImGuiToastType_Warning:
                return "Warning";
            case ImGuiToastType_Error:
                return "Error";
            case ImGuiToastType_Info:
                return "Info";
            }
        }

        return this->title;
    };

    inline const ImGuiToastType& get_type() { return this->type; };

    inline const ImVec4& get_color()
    {
        static const ImVec4 white = { 255, 255, 255, 255 };
        static const ImVec4 green = { 0, 255, 0, 255 };
        static const ImVec4 yellow = { 255, 255, 0, 255 };
        static const ImVec4 red = { 255, 0, 0, 255 };
        static const ImVec4 blue = { 0, 157, 255, 255 };

        switch (this->type)
        {
        case ImGuiToastType_None:
            return white;
        case ImGuiToastType_Success:
            return green;
        case ImGuiToastType_Warning:
            return yellow;
        case ImGuiToastType_Error:
            return red;
        case ImGuiToastType_Info:
            return blue;
        default:
            return white;
        }
    }

    inline const char* get_icon()
    {
        switch (this->type)
        {
        case ImGuiToastType_None:
            return NULL;
        case ImGuiToastType_Success:
            return ICON_FA_CIRCLE_CHECK;
        case ImGuiToastType_Warning:
            return ICON_FA_CIRCLE_EXCLAMATION;
        case ImGuiToastType_Error:
            return ICON_FA_CIRCLE_XMARK;
        case ImGuiToastType_Info:
            return ICON_FA_CIRCLE_INFO;
        default:
            return NULL;
        }
    }

    inline char* get_content() { return this->content; };

    inline uint64_t get_elapsed_time() { return get_tick_count() - this->creation_time; }

    inline const ImGuiToastPhase& get_phase()
    {
        const uint64_t elapsed = get_elapsed_time();

        if (elapsed > NOTIFY_FADE_IN_OUT_TIME + this->dismiss_time + NOTIFY_FADE_IN_OUT_TIME)
        {
            static const ImGuiToastPhase expired = ImGuiToastPhase_Expired;
            return expired;
        }
        else if (elapsed > NOTIFY_FADE_IN_OUT_TIME + this->dismiss_time)
        {
            static const ImGuiToastPhase fadeOut = ImGuiToastPhase_FadeOut;
            return fadeOut;
        }
        else if (elapsed > NOTIFY_FADE_IN_OUT_TIME)
        {
            static const ImGuiToastPhase wait = ImGuiToastPhase_Wait;
            return wait;
        }
        else
        {
            static const ImGuiToastPhase fadeIn = ImGuiToastPhase_FadeIn;
            return fadeIn;
        }
    }

    inline float get_fade_percent()
    {
        const uint64_t phase = get_phase();
        const uint64_t elapsed = get_elapsed_time();

        if (phase == ImGuiToastPhase_FadeIn)
        {
            return ((float)elapsed / (float)NOTIFY_FADE_IN_OUT_TIME) * NOTIFY_OPACITY;
        }
        else if (phase == ImGuiToastPhase_FadeOut)
        {
            return (1.f - (((float)elapsed - (float)NOTIFY_FADE_IN_OUT_TIME - (float)this->dismiss_time) / (float)NOTIFY_FADE_IN_OUT_TIME)) * NOTIFY_OPACITY;
        }

        return 1.f * NOTIFY_OPACITY;
    }

    inline static unsigned long long get_tick_count()
    {
        using namespace std::chrono;
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    }

public:
    // Constructors

    ImGuiToast(ImGuiToastType type, int dismiss_time = NOTIFY_DEFAULT_DISMISS)
    {
        IM_ASSERT(type < ImGuiToastType_COUNT);

        this->type = type;
        this->dismiss_time = dismiss_time;
        this->creation_time = get_tick_count();

        memset(this->title, 0, sizeof(this->title));
        memset(this->content, 0, sizeof(this->content));
    }

    ImGuiToast(ImGuiToastType type, const char* format, ...) : ImGuiToast(type) { NOTIFY_FORMAT(this->set_content, format); }

    ImGuiToast(ImGuiToastType type, int dismiss_time, const char* format, ...) : ImGuiToast(type, dismiss_time) { NOTIFY_FORMAT(this->set_content, format); }
};

namespace ImGui
{
    inline std::vector<ImGuiToast> notifications;

    /// <summary>
    /// Insert a new toast in the list
    /// </summary>
    inline void InsertNotification(const ImGuiToast& toast)
    {
        notifications.push_back(toast);
    }

    /// <summary>
    /// Remove a toast from the list by its index
    /// </summary>
    /// <param name="index">index of the toast to remove</param>
    inline void RemoveNotification(int index)
    {
        notifications.erase(notifications.begin() + index);
    }

    /// <summary>
    /// Render toasts, call at the end of your rendering!
    /// </summary>
    inline void RenderNotifications()
    {
        const auto vp_size = GetMainViewport()->Size;

        float height = 0.f;

        for (uint64_t i = 0; i < notifications.size(); i++)
        {
            auto* current_toast = &notifications[i];

            // Remove toast if expired
            if (current_toast->get_phase() == ImGuiToastPhase_Expired)
            {
                RemoveNotification(i);
                continue;
            }

            // Get icon, title and other data
            const auto icon = current_toast->get_icon();
            const auto title = current_toast->get_title();
            const auto content = current_toast->get_content();
            const auto default_title = current_toast->get_default_title();
            const auto opacity = current_toast->get_fade_percent(); // Get opacity based of the current phase

            // Window rendering
            auto text_color = current_toast->get_color();
            text_color.w = opacity;

            // Generate new unique name for this toast
            char window_name[50];
            snprintf(window_name, sizeof(window_name), "##TOAST%d", (int)i);

            //PushStyleColor(ImGuiCol_Text, text_color);
            SetNextWindowBgAlpha(opacity);
            SetNextWindowPos(ImVec2(vp_size.x - NOTIFY_PADDING_X, vp_size.y - NOTIFY_PADDING_Y - height), ImGuiCond_Always, ImVec2(1.0f, 1.0f));
            Begin(window_name, NULL, NOTIFY_TOAST_FLAGS);

            // Here we render the toast content
            {
                PushTextWrapPos(vp_size.x / 3.f); // We want to support multi-line text, this will wrap the text after 1/3 of the screen width

                bool was_title_rendered = false;

                // If an icon is set
                if (!NOTIFY_NULL_OR_EMPTY(icon))
                {
                    // {icon); // Render icon text
                    TextColored(text_color, "%s", icon);
                    was_title_rendered = true;
                }

                // If a title is set
                if (!NOTIFY_NULL_OR_EMPTY(title))
                {
                    // If a title and an icon is set, we want to render on same line
                    if (!NOTIFY_NULL_OR_EMPTY(icon))
                        SameLine();

                    Text("%s", title); // Render title text
                    was_title_rendered = true;
                }
                else if (!NOTIFY_NULL_OR_EMPTY(default_title))
                {
                    if (!NOTIFY_NULL_OR_EMPTY(icon))
                        SameLine();

                    Text("%s", default_title); // Render default title text (ImGuiToastType_Success -> "Success", etc...)
                    was_title_rendered = true;
                }

                // In case ANYTHING was rendered in the top, we want to add a small padding so the text (or icon) looks centered vertically
                if (was_title_rendered && !NOTIFY_NULL_OR_EMPTY(content))
                {
                    SetCursorPosY(GetCursorPosY() + 5.f); // Must be a better way to do this!!!!
                }

                // If a content is set
                if (!NOTIFY_NULL_OR_EMPTY(content))
                {
                    if (was_title_rendered)
                    {
                        Separator();
                    }

                    Text("%s", content); // Render content text
                }

                PopTextWrapPos();
            }

            // Save height for next toasts
            height += GetWindowHeight() + NOTIFY_PADDING_MESSAGE_Y;

            // End
            End();
        }
    }
}

#endif
