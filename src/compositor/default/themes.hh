#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_THEMES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_THEMES_HH

#include "jetstream/render/sakura/color.hh"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace Jetstream {

inline const std::unordered_map<std::string, Sakura::Palette> themes = {
    {
        "Dark", {
            // Core Brand Colors
            {"cyber_blue", {0.30f, 0.75f, 0.95f, 1.0f}},
            {"accent_color", {0.90f, 0.65f, 0.50f, 1.00f}},
            {"accent_active", {1.00f, 0.85f, 0.70f, 1.00f}},

            // Welcome Card Icon Colors
            {"welcome_icon_new", {0.30f, 0.75f, 0.95f, 1.0f}},
            {"welcome_icon_open", {0.30f, 0.75f, 0.95f, 1.0f}},
            {"welcome_icon_examples", {0.30f, 0.75f, 0.95f, 1.0f}},

            // Status Colors
            {"success_green", {0.0f, 1.0f, 0.0f, 1.0f}},
            {"warning_yellow", {1.0f, 0.8f, 0.0f, 1.0f}},
            {"error_red", {1.0f, 0.0f, 0.0f, 1.0f}},
            {"info_blue", {0.0f, 0.0f, 1.0f, 1.0f}},

            // Background Colors
            {"background", {0.0f, 0.0f, 0.0f, 1.00f}},
            {"panel", {0.05f, 0.05f, 0.05f, 1.00f}},
            {"card", {0.08f, 0.08f, 0.08f, 1.00f}},
            {"popup_bg", {0.06f, 0.06f, 0.06f, 1.0f}},
            {"modal_dim", {0.00f, 0.00f, 0.00f, 0.60f}},
            {"notification_bg", {0.08f, 0.08f, 0.08f, 0.90f}},

            // Button Colors
            {"action_btn", {0.20f, 0.47f, 0.96f, 1.0f}},
            {"action_btn_hovered", {0.25f, 0.52f, 0.99f, 1.0f}},
            {"action_btn_active", {0.15f, 0.35f, 0.85f, 1.0f}},

            // Text Colors
            {"text_primary", {0.90f, 0.90f, 0.90f, 1.00f}},
            {"text_secondary", {0.50f, 0.50f, 0.52f, 1.00f}},
            {"text_disabled", {0.40f, 0.40f, 0.40f, 1.0f}},
            {"text_selected_bg", {0.25f, 0.25f, 0.27f, 0.50f}},

            // Border Colors
            {"border", {0.18f, 0.18f, 0.18f, 0.75f}},
            {"border_shadow", {0.00f, 0.00f, 0.00f, 0.00f}},

            // Frame Colors
            {"frame_bg_hovered", {0.13f, 0.13f, 0.13f, 0.95f}},
            {"frame_bg_active", {0.17f, 0.17f, 0.17f, 0.95f}},

            // Title Bar Colors
            {"title_bg_active", {0.06f, 0.06f, 0.06f, 0.97f}},
            {"title_bg_collapsed", {0.04f, 0.04f, 0.04f, 0.80f}},

            // Scrollbar Colors
            {"scrollbar_bg", {0.0f, 0.0f, 0.0f, 0.0f}},
            {"scrollbar_grab", {0.20f, 0.20f, 0.20f, 0.80f}},
            {"scrollbar_grab_hovered", {0.28f, 0.28f, 0.28f, 0.80f}},
            {"scrollbar_grab_active", {0.35f, 0.35f, 0.35f, 0.80f}},

            // Button Colors
            {"button", {0.10f, 0.10f, 0.10f, 1.00f}},
            {"button_hovered", {0.12f, 0.12f, 0.12f, 1.00f}},
            {"button_active", {0.14f, 0.14f, 0.14f, 1.00f}},

            // Header Colors
            {"header", {0.10f, 0.10f, 0.10f, 0.80f}},
            {"header_hovered", {0.15f, 0.15f, 0.15f, 0.80f}},
            {"header_active", {0.20f, 0.20f, 0.20f, 0.80f}},

            // Separator Colors
            {"separator", {0.15f, 0.15f, 0.15f, 0.50f}},
            {"separator_hovered", {0.25f, 0.25f, 0.25f, 0.80f}},
            {"separator_active", {0.35f, 0.35f, 0.35f, 1.00f}},

            // Resize Grip Colors
            {"resize_grip", {0.18f, 0.18f, 0.18f, 0.60f}},
            {"resize_grip_hovered", {0.28f, 0.28f, 0.28f, 0.80f}},
            {"resize_grip_active", {0.38f, 0.38f, 0.38f, 1.00f}},

            // Tab Colors
            {"tab", {0.04f, 0.04f, 0.04f, 0.95f}},
            {"tab_hovered", {0.12f, 0.12f, 0.12f, 1.00f}},
            {"tab_selected", {0.08f, 0.08f, 0.08f, 0.95f}},
            {"tab_dimmed", {0.03f, 0.03f, 0.03f, 0.95f}},
            {"tab_dimmed_selected", {0.06f, 0.06f, 0.06f, 0.95f}},

            // Docking Colors
            {"docking_preview", {0.30f, 0.30f, 0.30f, 0.50f}},
            {"docking_empty_bg", {0.02f, 0.02f, 0.02f, 1.00f}},

            // Plot Colors
            {"plot_lines", {0.50f, 0.50f, 0.52f, 1.00f}},
            {"plot_lines_hovered", {0.60f, 0.60f, 0.62f, 1.00f}},
            {"plot_histogram", {0.40f, 0.40f, 0.42f, 1.00f}},
            {"plot_histogram_hovered", {0.50f, 0.50f, 0.52f, 1.00f}},

            // Table Colors
            {"table_header_bg", {0.06f, 0.06f, 0.06f, 0.95f}},
            {"table_border_strong", {0.15f, 0.15f, 0.15f, 0.70f}},
            {"table_border_light", {0.10f, 0.10f, 0.10f, 0.50f}},
            {"table_row_bg", {0.00f, 0.00f, 0.00f, 0.00f}},
            {"table_row_bg_alt", {1.00f, 1.00f, 1.00f, 0.02f}},

            // Drag & Drop Colors
            {"drag_drop_target", {0.50f, 0.50f, 0.52f, 0.90f}},

            // Navigation Colors
            {"nav_windowing_highlight", {1.00f, 1.00f, 1.00f, 0.50f}},
            {"nav_windowing_dim_bg", {0.00f, 0.00f, 0.00f, 0.60f}},

            // Node Editor Colors
            {"cell_background", {0.03f, 0.03f, 0.03f, 0.90f}},
            {"node_background", {0.05f, 0.05f, 0.05f, 1.0f}},
            {"node_outline", {0.878f, 0.573f, 0.0f, 1.0f}},
            {"node_outline_error", {0.86f, 0.24f, 0.24f, 1.0f}},
            {"node_outline_pending", {0.50f, 0.50f, 0.50f, 1.0f}},
            {"node_title_bar", {0.0f, 0.0f, 0.0f, 0.0f}},
            {"node_pin", {0.878f, 0.573f, 0.0f, 1.0f}},
            {"node_link", {0.878f, 0.573f, 0.0f, 1.0f}},
            {"grid_line", {0.12f, 0.12f, 0.13f, 1.0f}},
            {"grid_background", {0.0f, 0.0f, 0.0f, 0.0f}},

            // Debug Colors
            {"debug_red", {1.0f, 0.0f, 0.0f, 1.0f}},
            {"debug_green", {0.0f, 1.0f, 0.0f, 1.0f}},
            {"debug_blue", {0.0f, 0.0f, 1.0f, 1.0f}},
            {"debug_yellow", {1.0f, 1.0f, 0.0f, 1.0f}},
            {"debug_white", {1.0f, 1.0f, 1.0f, 1.0f}},
            {"debug_black", {0.0f, 0.0f, 0.0f, 1.0f}},

            // Selection Colors
            {"selection_lime", {0.5f, 1.0f, 0.0f, 1.0f}},
        }
    },
    {
        "Light", {
            // Core Brand Colors
            {"cyber_blue", {0.15f, 0.55f, 0.85f, 1.0f}},
            {"accent_color", {0.80f, 0.45f, 0.25f, 1.00f}},
            {"accent_active", {0.90f, 0.60f, 0.40f, 1.00f}},

            // Welcome Card Icon Colors
            {"welcome_icon_new", {0.15f, 0.55f, 0.85f, 1.0f}},
            {"welcome_icon_open", {0.15f, 0.55f, 0.85f, 1.0f}},
            {"welcome_icon_examples", {0.15f, 0.55f, 0.85f, 1.0f}},

            // Status Colors
            {"success_green", {0.0f, 0.6f, 0.0f, 1.0f}},
            {"warning_yellow", {0.95f, 0.90f, 0.0f, 1.0f}},
            {"error_red", {0.9f, 0.15f, 0.15f, 1.0f}},
            {"info_blue", {0.15f, 0.4f, 0.9f, 1.0f}},

            // Background Colors
            {"background", {0.97f, 0.97f, 0.98f, 1.00f}},
            {"panel", {0.95f, 0.95f, 0.96f, 1.00f}},
            {"card", {0.93f, 0.93f, 0.94f, 1.00f}},
            {"popup_bg", {0.96f, 0.96f, 0.97f, 1.0f}},
            {"modal_dim", {0.00f, 0.00f, 0.00f, 0.35f}},
            {"notification_bg", {0.96f, 0.96f, 0.97f, 0.90f}},

            // Button Colors
            {"action_btn", {0.20f, 0.47f, 0.96f, 1.0f}},
            {"action_btn_hovered", {0.25f, 0.52f, 0.99f, 1.0f}},
            {"action_btn_active", {0.15f, 0.35f, 0.85f, 1.0f}},

            // Text Colors
            {"text_primary", {0.12f, 0.12f, 0.14f, 1.00f}},
            {"text_secondary", {0.45f, 0.45f, 0.48f, 1.00f}},
            {"text_disabled", {0.65f, 0.65f, 0.68f, 1.0f}},
            {"text_selected_bg", {0.70f, 0.70f, 0.72f, 0.50f}},

            // Border Colors
            {"border", {0.78f, 0.78f, 0.80f, 0.75f}},
            {"border_shadow", {0.00f, 0.00f, 0.00f, 0.00f}},

            // Frame Colors
            {"frame_bg_hovered", {0.86f, 0.86f, 0.87f, 0.95f}},
            {"frame_bg_active", {0.80f, 0.80f, 0.82f, 0.95f}},

            // Title Bar Colors
            {"title_bg_active", {0.94f, 0.94f, 0.95f, 0.97f}},
            {"title_bg_collapsed", {0.96f, 0.96f, 0.97f, 0.80f}},

            // Scrollbar Colors
            {"scrollbar_bg", {0.0f, 0.0f, 0.0f, 0.0f}},
            {"scrollbar_grab", {0.65f, 0.65f, 0.67f, 0.80f}},
            {"scrollbar_grab_hovered", {0.55f, 0.55f, 0.57f, 0.80f}},
            {"scrollbar_grab_active", {0.45f, 0.45f, 0.47f, 0.80f}},

            // Button Colors
            {"button", {0.86f, 0.86f, 0.88f, 1.00f}},
            {"button_hovered", {0.82f, 0.82f, 0.84f, 1.00f}},
            {"button_active", {0.78f, 0.78f, 0.80f, 1.00f}},

            // Header Colors
            {"header", {0.90f, 0.90f, 0.91f, 0.80f}},
            {"header_hovered", {0.85f, 0.85f, 0.86f, 0.80f}},
            {"header_active", {0.80f, 0.80f, 0.81f, 0.80f}},

            // Separator Colors
            {"separator", {0.80f, 0.80f, 0.82f, 0.50f}},
            {"separator_hovered", {0.70f, 0.70f, 0.72f, 0.80f}},
            {"separator_active", {0.60f, 0.60f, 0.62f, 1.00f}},

            // Resize Grip Colors
            {"resize_grip", {0.75f, 0.75f, 0.77f, 0.60f}},
            {"resize_grip_hovered", {0.65f, 0.65f, 0.67f, 0.80f}},
            {"resize_grip_active", {0.55f, 0.55f, 0.57f, 1.00f}},

            // Tab Colors
            {"tab", {0.92f, 0.92f, 0.93f, 0.95f}},
            {"tab_hovered", {0.88f, 0.88f, 0.89f, 1.00f}},
            {"tab_selected", {0.97f, 0.97f, 0.98f, 0.95f}},
            {"tab_dimmed", {0.90f, 0.90f, 0.91f, 0.95f}},
            {"tab_dimmed_selected", {0.94f, 0.94f, 0.95f, 0.95f}},

            // Docking Colors
            {"docking_preview", {0.60f, 0.60f, 0.62f, 0.50f}},
            {"docking_empty_bg", {0.94f, 0.94f, 0.95f, 1.00f}},

            // Plot Colors
            {"plot_lines", {0.45f, 0.45f, 0.47f, 1.00f}},
            {"plot_lines_hovered", {0.35f, 0.35f, 0.37f, 1.00f}},
            {"plot_histogram", {0.55f, 0.55f, 0.57f, 1.00f}},
            {"plot_histogram_hovered", {0.45f, 0.45f, 0.47f, 1.00f}},

            // Table Colors
            {"table_header_bg", {0.94f, 0.94f, 0.95f, 0.95f}},
            {"table_border_strong", {0.80f, 0.80f, 0.82f, 0.70f}},
            {"table_border_light", {0.88f, 0.88f, 0.90f, 0.50f}},
            {"table_row_bg", {0.00f, 0.00f, 0.00f, 0.00f}},
            {"table_row_bg_alt", {0.00f, 0.00f, 0.00f, 0.02f}},

            // Drag & Drop Colors
            {"drag_drop_target", {0.45f, 0.45f, 0.47f, 0.90f}},

            // Navigation Colors
            {"nav_windowing_highlight", {0.00f, 0.00f, 0.00f, 0.50f}},
            {"nav_windowing_dim_bg", {0.00f, 0.00f, 0.00f, 0.60f}},

            // Node Editor Colors
            {"cell_background", {0.95f, 0.95f, 0.96f, 0.90f}},
            {"node_config_background", {0.90f, 0.90f, 0.91f, 1.0f}},
            {"node_background", {1.00f, 1.00f, 1.00f, 1.0f}},
            {"node_outline", {0.80f, 0.55f, 0.05f, 1.0f}},
            {"node_outline_error", {0.86f, 0.24f, 0.24f, 1.0f}},
            {"node_outline_pending", {0.55f, 0.55f, 0.55f, 1.0f}},
            {"node_title_bar", {0.0f, 0.0f, 0.0f, 0.0f}},
            {"node_pin", {0.80f, 0.55f, 0.05f, 1.0f}},
            {"node_link", {0.80f, 0.55f, 0.05f, 1.0f}},
            {"grid_line", {0.82f, 0.82f, 0.83f, 1.0f}},
            {"grid_background", {0.0f, 0.0f, 0.0f, 0.0f}},

            // Debug Colors
            {"debug_red", {1.0f, 0.0f, 0.0f, 1.0f}},
            {"debug_green", {0.0f, 1.0f, 0.0f, 1.0f}},
            {"debug_blue", {0.0f, 0.0f, 1.0f, 1.0f}},
            {"debug_yellow", {1.0f, 1.0f, 0.0f, 1.0f}},
            {"debug_white", {1.0f, 1.0f, 1.0f, 1.0f}},
            {"debug_black", {0.0f, 0.0f, 0.0f, 1.0f}},

            // Selection Colors
            {"selection_lime", {0.4f, 0.8f, 0.0f, 1.0f}},
        }
    }
};

inline std::vector<std::string> BuildThemeKeys() {
    std::vector<std::string> themeKeys;
    themeKeys.reserve(themes.size());
    for (const auto& [themeKey, _] : themes) {
        themeKeys.push_back(themeKey);
    }
    std::sort(themeKeys.begin(), themeKeys.end());
    return themeKeys;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_THEMES_HH
