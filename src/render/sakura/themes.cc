#include "themes.hh"

#include <algorithm>
#include <cctype>

namespace Jetstream::Sakura {

const std::unordered_map<std::string, Palette> Themes = {
    {
        "Dark", {
            // Core Brand Colors
            {"cyber_blue", {0.30f, 0.75f, 0.95f, 1.0f}},
            {"accent_color", {0.90f, 0.65f, 0.50f, 1.00f}},

            // Welcome Card Colors
            {"card_surface", {0.09f, 0.09f, 0.09f, 1.00f}},
            {"card_surface_hovered", {0.13f, 0.13f, 0.13f, 1.00f}},

            // Gallery Accent Colors
            {"gallery_green", {0.329f, 0.800f, 0.522f, 1.0f}},
            {"gallery_lime", {0.580f, 0.765f, 0.325f, 1.0f}},
            {"gallery_blue", {0.494f, 0.702f, 0.996f, 1.0f}},
            {"gallery_cyan", {0.071f, 0.784f, 0.847f, 1.0f}},
            {"gallery_orange", {0.984f, 0.576f, 0.310f, 1.0f}},
            {"gallery_amber", {0.902f, 0.639f, 0.141f, 1.0f}},
            {"gallery_coral", {1.000f, 0.545f, 0.506f, 1.0f}},
            {"gallery_teal", {0.373f, 0.820f, 0.706f, 1.0f}},
            {"gallery_purple", {0.800f, 0.584f, 0.976f, 1.0f}},

            // Status Colors
            {"success_green", {0.0f, 1.0f, 0.0f, 1.0f}},
            {"status_success", {0.10f, 0.50f, 0.27f, 1.0f}},
            {"status_success_border", {0.20f, 0.68f, 0.40f, 1.0f}},
            {"status_warning", {0.58f, 0.38f, 0.06f, 1.0f}},
            {"status_warning_border", {0.78f, 0.53f, 0.12f, 1.0f}},
            {"warning_yellow", {1.0f, 0.8f, 0.0f, 1.0f}},
            {"favorite_star", {1.0f, 0.76f, 0.22f, 1.0f}},
            {"error_red", {1.0f, 0.0f, 0.0f, 1.0f}},
            {"progress_warning", {0.56f, 0.40f, 0.10f, 1.0f}},
            {"progress_error", {0.56f, 0.18f, 0.18f, 1.0f}},
            {"info_blue", {0.35f, 0.62f, 1.0f, 1.0f}},
            {"run_state_running", {0.36f, 0.86f, 0.52f, 1.0f}},
            {"run_state_paused", {1.0f, 0.62f, 0.26f, 1.0f}},

            // Banner Colors
            {"banner_info_bg", {0.30f, 0.55f, 1.00f, 0.16f}},
            {"banner_info_border", {0.30f, 0.55f, 1.00f, 0.55f}},
            {"banner_info_text", {0.62f, 0.78f, 1.00f, 1.00f}},
            {"banner_warning_bg", {1.00f, 0.80f, 0.00f, 0.10f}},
            {"banner_warning_border", {1.00f, 0.80f, 0.00f, 0.35f}},
            {"banner_warning_text", {0.98f, 0.90f, 0.70f, 1.00f}},
            {"tint_green_hover", {0.25f, 0.95f, 0.45f, 0.22f}},
            {"tint_green_text", {0.45f, 1.00f, 0.60f, 1.00f}},
            {"tint_red_bg", {1.00f, 0.25f, 0.32f, 0.14f}},
            {"tint_red_hover", {1.00f, 0.25f, 0.32f, 0.24f}},
            {"tint_red_border", {1.00f, 0.25f, 0.32f, 0.65f}},
            {"tint_red_text", {1.00f, 0.42f, 0.46f, 1.00f}},
            {"banner_success_bg", {0.30f, 0.85f, 0.45f, 0.10f}},
            {"banner_success_border", {0.30f, 0.85f, 0.45f, 0.35f}},
            {"banner_success_text", {0.55f, 0.92f, 0.66f, 1.00f}},
            {"banner_error_bg", {0.95f, 0.35f, 0.35f, 0.10f}},
            {"banner_error_border", {0.95f, 0.35f, 0.35f, 0.35f}},
            {"banner_error_text", {0.98f, 0.82f, 0.82f, 1.00f}},

            // Background Colors
            {"background", {0.0f, 0.0f, 0.0f, 1.00f}},
            {"workspace_particle", {0.976f, 0.976f, 0.976f, 1.0f}},
            {"card", {0.08f, 0.08f, 0.08f, 1.00f}},
            {"agent_composer", {0.10f, 0.10f, 0.10f, 1.00f}},
            {"agent_outline", {1.00f, 1.00f, 1.00f, 0.12f}},
            {"agent_activity", {0.20f, 0.47f, 0.96f, 1.00f}},
            {"popup_bg", {0.06f, 0.06f, 0.06f, 1.0f}},
            {"modal_dim", {0.00f, 0.00f, 0.00f, 0.60f}},
            {"notification_bg", {0.08f, 0.08f, 0.08f, 0.90f}},
            {"pill_field_bg", {0.08f, 0.08f, 0.08f, 0.97f}},
            {"plot_grid", {0.12f, 0.12f, 0.12f, 1.0f}},
            {"plot_grid_major", {0.50f, 0.50f, 0.50f, 1.0f}},
            {"plot_accent", {1.00f, 0.85f, 0.00f, 1.0f}},
            {"plot_halo", {0.00f, 0.00f, 0.00f, 0.65f}},
            {"transparent", {0.00f, 0.00f, 0.00f, 0.00f}},
            {"remote_btn", {0.13f, 0.62f, 0.30f, 1.0f}},
            {"remote_btn_hovered", {0.16f, 0.68f, 0.34f, 1.0f}},

            // Button Colors
            {"action_btn", {0.20f, 0.47f, 0.96f, 1.0f}},
            {"action_btn_hovered", {0.25f, 0.52f, 0.99f, 1.0f}},
            {"action_btn_active", {0.15f, 0.35f, 0.85f, 1.0f}},
            {"action_btn_text", {1.00f, 1.00f, 1.00f, 1.0f}},
            {"action_btn_outline", {0.35f, 0.65f, 1.00f, 0.85f}},
            {"action_btn_highlight", {1.00f, 1.00f, 1.00f, 0.30f}},
            {"destructive_btn", {0.60f, 0.00f, 0.00f, 1.0f}},
            {"destructive_btn_hovered", {0.70f, 0.00f, 0.00f, 1.0f}},
            {"destructive_btn_active", {0.50f, 0.00f, 0.00f, 1.0f}},
            {"destructive_btn_text", {1.00f, 1.00f, 1.00f, 1.0f}},
            {"destructive_btn_outline", {0.86f, 0.24f, 0.24f, 1.0f}},
            {"contrast_btn", {0.93f, 0.93f, 0.93f, 1.0f}},
            {"contrast_btn_hovered", {0.82f, 0.82f, 0.82f, 1.0f}},
            {"contrast_btn_active", {0.70f, 0.70f, 0.70f, 1.0f}},
            {"contrast_btn_text", {0.08f, 0.08f, 0.08f, 1.0f}},
            {"warning_btn", {0.85f, 0.45f, 0.10f, 1.0f}},
            {"warning_btn_hovered", {0.95f, 0.55f, 0.15f, 1.0f}},
            {"warning_btn_active", {0.75f, 0.35f, 0.05f, 1.0f}},
            {"warning_btn_text", {1.00f, 1.00f, 1.00f, 1.0f}},
            {"warning_btn_outline", {1.00f, 0.65f, 0.25f, 0.85f}},

            // Text Colors
            {"text_primary", {0.90f, 0.90f, 0.90f, 1.00f}},
            {"text_secondary", {0.50f, 0.50f, 0.52f, 1.00f}},
            {"text_disabled", {0.40f, 0.40f, 0.40f, 1.0f}},
            {"text_selected_bg", {0.25f, 0.25f, 0.27f, 0.50f}},

            // Editor Colors (Ayu Dark inspired)
            {"editor_background", {0.05f, 0.05f, 0.05f, 1.00f}},
            {"editor_text", {0.70f, 0.69f, 0.68f, 1.00f}},
            {"editor_comment", {0.36f, 0.40f, 0.45f, 1.00f}},
            {"editor_keyword", {1.00f, 0.56f, 0.25f, 1.00f}},
            {"editor_string", {0.67f, 0.85f, 0.30f, 1.00f}},
            {"editor_constant", {0.82f, 0.65f, 1.00f, 1.00f}},
            {"editor_function", {1.00f, 0.82f, 0.45f, 1.00f}},
            {"editor_type", {0.45f, 0.82f, 1.00f, 1.00f}},
            {"editor_console_background", {0.03f, 0.03f, 0.04f, 0.97f}},
            {"editor_console_header_outline", {0.20f, 0.20f, 0.23f, 1.00f}},
            {"editor_status_info", {0.10f, 0.26f, 0.42f, 1.00f}},
            {"editor_status_info_text", {0.84f, 0.93f, 1.00f, 1.00f}},
            {"editor_status_info_separator", {0.08f, 0.21f, 0.34f, 1.00f}},
            {"editor_status_success", {0.14f, 0.36f, 0.24f, 1.00f}},
            {"editor_status_success_text", {0.84f, 1.00f, 0.90f, 1.00f}},
            {"editor_status_success_separator", {0.11f, 0.29f, 0.19f, 1.00f}},
            {"editor_status_warning", {0.48f, 0.34f, 0.10f, 1.00f}},
            {"editor_status_warning_text", {1.00f, 0.94f, 0.78f, 1.00f}},
            {"editor_status_warning_separator", {0.38f, 0.27f, 0.08f, 1.00f}},
            {"editor_status_error", {0.45f, 0.16f, 0.16f, 1.00f}},
            {"editor_status_error_text", {1.00f, 0.86f, 0.86f, 1.00f}},
            {"editor_status_error_separator", {0.36f, 0.13f, 0.13f, 1.00f}},
            {"editor_selection", {0.153f, 0.216f, 0.278f, 0.72f}},
            {"editor_selection_match", {0.42f, 0.32f, 0.12f, 0.55f}},
            {"editor_active_line", {0.12f, 0.14f, 0.18f, 0.40f}},
            {"editor_line_number", {0.24f, 0.26f, 0.30f, 1.00f}},
            {"editor_cursor", {0.93f, 0.76f, 0.46f, 1.00f}},
            {"editor_gutter_separator", {0.18f, 0.19f, 0.22f, 0.35f}},
            {"editor_scrollbar_track", {0.12f, 0.12f, 0.13f, 1.00f}},
            {"editor_scrollbar_thumb", {0.42f, 0.42f, 0.42f, 0.80f}},

            // Markdown Callout Colors
            {"callout_note", {0.27f, 0.58f, 0.97f, 1.00f}},
            {"callout_tip", {0.25f, 0.73f, 0.31f, 1.00f}},
            {"callout_important", {0.67f, 0.49f, 0.97f, 1.00f}},
            {"callout_warning", {0.82f, 0.60f, 0.13f, 1.00f}},
            {"callout_caution", {0.97f, 0.32f, 0.29f, 1.00f}},
            {"callout_error", {1.00f, 0.48f, 0.45f, 1.00f}},

            // Markdown Code Block Colors
            {"markdown_code_block", {0.09f, 0.09f, 0.09f, 1.00f}},
            {"markdown_code_block_border", {0.20f, 0.20f, 0.20f, 1.00f}},

            // Chat Colors
            {"chat_user_bubble", {0.20f, 0.47f, 0.96f, 1.00f}},
            {"chat_user_bubble_outline", {0.35f, 0.65f, 1.00f, 0.85f}},

            // Voice Colors
            {"voice_bg", {0.035f, 0.035f, 0.04f, 1.0f}},
            {"voice_listening", {0.35f, 0.65f, 1.00f, 1.0f}},
            {"voice_busy", {1.00f, 0.58f, 0.32f, 1.0f}},
            {"voice_dim", {0.55f, 0.55f, 0.60f, 1.0f}},
            {"voice_stop", {0.92f, 0.30f, 0.28f, 1.0f}},

            // Border Colors
            {"border", {0.18f, 0.18f, 0.18f, 0.75f}},

            // Frame Colors
            {"frame_bg_hovered", {0.13f, 0.13f, 0.13f, 0.95f}},
            {"frame_bg_active", {0.17f, 0.17f, 0.17f, 0.95f}},

            // Scrollbar Colors
            {"scrollbar_bg", {0.0f, 0.0f, 0.0f, 0.0f}},
            {"scrollbar_grab", {0.20f, 0.20f, 0.20f, 0.80f}},

            // Button Colors
            {"button", {0.10f, 0.10f, 0.10f, 1.00f}},
            {"button_hovered", {0.12f, 0.12f, 0.12f, 1.00f}},
            {"button_active", {0.14f, 0.14f, 0.14f, 1.00f}},
            {"button_text", {0.90f, 0.90f, 0.90f, 1.00f}},
            {"button_outline", {0.18f, 0.18f, 0.18f, 0.75f}},

            // Header Colors
            {"header", {0.10f, 0.10f, 0.10f, 0.80f}},
            {"header_hovered", {0.15f, 0.15f, 0.15f, 0.80f}},
            {"header_active", {0.20f, 0.20f, 0.20f, 0.80f}},

            // Separator Colors
            {"separator", {0.15f, 0.15f, 0.15f, 0.50f}},
            {"separator_hovered", {0.25f, 0.25f, 0.25f, 0.80f}},

            // Resize Grip Colors
            {"resize_grip", {0.18f, 0.18f, 0.18f, 0.60f}},
            {"resize_grip_hovered", {0.28f, 0.28f, 0.28f, 0.80f}},
            {"resize_grip_active", {0.38f, 0.38f, 0.38f, 1.00f}},

            // Window Chrome Colors
            {"tile_bg", {0.055f, 0.055f, 0.06f, 1.00f}},
            {"tile_border", {1.00f, 1.00f, 1.00f, 0.08f}},
            {"tile_header_text", {0.68f, 0.68f, 0.70f, 1.00f}},
            {"tile_button_hover", {0.97f, 0.97f, 0.98f, 1.00f}},
            {"tile_button_active_bg", {0.30f, 0.75f, 0.95f, 0.18f}},
            {"tile_shadow", {0.00f, 0.00f, 0.00f, 0.55f}},
            {"drop_fill", {0.24f, 0.52f, 0.98f, 0.18f}},
            {"drop_border", {0.30f, 0.58f, 1.00f, 0.95f}},
            {"pill_bg", {0.06f, 0.06f, 0.065f, 1.00f}},
            {"pill_hover_bg", {0.10f, 0.10f, 0.11f, 1.00f}},
            {"pill_active_bg", {0.16f, 0.16f, 0.175f, 1.00f}},
            {"pill_selected_bg", {0.19f, 0.19f, 0.205f, 1.00f}},
            {"pill_text", {0.88f, 0.88f, 0.90f, 1.00f}},
            {"chrome_button_hover", {1.00f, 1.00f, 1.00f, 0.10f}},

            // Settings Colors
            {"setting_card_border", {1.00f, 1.00f, 1.00f, 0.08f}},
            {"setting_card_hovered", {1.00f, 1.00f, 1.00f, 0.04f}},
            {"primary_btn", {0.96f, 0.96f, 0.96f, 1.00f}},
            {"primary_btn_hovered", {0.86f, 0.86f, 0.86f, 1.00f}},
            {"primary_btn_text", {0.06f, 0.06f, 0.06f, 1.00f}},
            {"toggle_on", {0.25f, 0.47f, 0.95f, 1.00f}},
            {"toggle_on_hovered", {0.32f, 0.53f, 0.98f, 1.00f}},
            {"toggle_off", {0.26f, 0.26f, 0.26f, 1.00f}},
            {"toggle_off_hovered", {0.31f, 0.31f, 0.31f, 1.00f}},
            {"toggle_knob", {1.00f, 1.00f, 1.00f, 1.00f}},

            // Table Colors
            {"table_header_bg", {0.06f, 0.06f, 0.06f, 0.95f}},
            {"table_border_strong", {0.15f, 0.15f, 0.15f, 0.70f}},
            {"table_border_light", {0.10f, 0.10f, 0.10f, 0.50f}},
            {"table_row_bg", {0.00f, 0.00f, 0.00f, 0.00f}},
            {"table_row_bg_alt", {1.00f, 1.00f, 1.00f, 0.02f}},

            // Node Editor Colors
            {"node_background", {0.05f, 0.05f, 0.05f, 1.0f}},
            {"node_outline", {0.878f, 0.573f, 0.0f, 1.0f}},
            {"node_outline_error", {0.86f, 0.24f, 0.24f, 1.0f}},
            {"node_outline_pending", {0.50f, 0.50f, 0.50f, 1.0f}},
            {"node_pin", {0.878f, 0.573f, 0.0f, 1.0f}},
            {"node_link", {0.878f, 0.573f, 0.0f, 1.0f}},
            {"node_toggle_on", {0.40f, 0.80f, 0.40f, 1.0f}},
            {"grid_line", {0.12f, 0.12f, 0.13f, 1.0f}},

        }
    },
    {
        "Light", {
            // Core Brand Colors
            {"cyber_blue", {0.15f, 0.55f, 0.85f, 1.0f}},
            {"accent_color", {0.80f, 0.45f, 0.25f, 1.00f}},

            // Welcome Card Colors
            {"card_surface", {1.00f, 1.00f, 1.00f, 1.00f}},
            {"card_surface_hovered", {0.94f, 0.94f, 0.95f, 1.00f}},

            // Gallery Accent Colors
            {"gallery_green", {0.10f, 0.52f, 0.28f, 1.0f}},
            {"gallery_lime", {0.33f, 0.48f, 0.08f, 1.0f}},
            {"gallery_blue", {0.16f, 0.38f, 0.80f, 1.0f}},
            {"gallery_cyan", {0.02f, 0.46f, 0.53f, 1.0f}},
            {"gallery_orange", {0.76f, 0.34f, 0.07f, 1.0f}},
            {"gallery_amber", {0.62f, 0.40f, 0.02f, 1.0f}},
            {"gallery_coral", {0.80f, 0.27f, 0.23f, 1.0f}},
            {"gallery_teal", {0.07f, 0.50f, 0.42f, 1.0f}},
            {"gallery_purple", {0.50f, 0.27f, 0.78f, 1.0f}},

            // Status Colors
            {"success_green", {0.0f, 0.6f, 0.0f, 1.0f}},
            {"status_success", {0.30f, 0.76f, 0.46f, 1.0f}},
            {"status_success_border", {0.22f, 0.66f, 0.38f, 1.0f}},
            {"status_warning", {0.93f, 0.66f, 0.20f, 1.0f}},
            {"status_warning_border", {0.82f, 0.55f, 0.12f, 1.0f}},
            {"warning_yellow", {0.64f, 0.38f, 0.0f, 1.0f}},
            {"favorite_star", {0.78f, 0.50f, 0.00f, 1.0f}},
            {"error_red", {0.9f, 0.15f, 0.15f, 1.0f}},
            {"progress_warning", {0.98f, 0.78f, 0.45f, 1.0f}},
            {"progress_error", {0.95f, 0.60f, 0.60f, 1.0f}},
            {"info_blue", {0.15f, 0.4f, 0.9f, 1.0f}},
            {"run_state_running", {0.13f, 0.60f, 0.32f, 1.0f}},
            {"run_state_paused", {0.88f, 0.45f, 0.08f, 1.0f}},

            // Banner Colors
            {"banner_info_bg", {0.15f, 0.40f, 0.90f, 0.08f}},
            {"banner_info_border", {0.15f, 0.40f, 0.90f, 0.35f}},
            {"banner_info_text", {0.08f, 0.25f, 0.60f, 1.00f}},
            {"banner_warning_bg", {0.80f, 0.55f, 0.00f, 0.10f}},
            {"banner_warning_border", {0.80f, 0.55f, 0.00f, 0.40f}},
            {"banner_warning_text", {0.55f, 0.35f, 0.05f, 1.00f}},
            {"tint_green_hover", {0.10f, 0.65f, 0.30f, 0.14f}},
            {"tint_green_text", {0.05f, 0.50f, 0.22f, 1.00f}},
            {"tint_red_bg", {0.90f, 0.15f, 0.20f, 0.08f}},
            {"tint_red_hover", {0.90f, 0.15f, 0.20f, 0.14f}},
            {"tint_red_border", {0.90f, 0.15f, 0.20f, 0.50f}},
            {"tint_red_text", {0.80f, 0.10f, 0.15f, 1.00f}},
            {"banner_success_bg", {0.10f, 0.60f, 0.30f, 0.08f}},
            {"banner_success_border", {0.10f, 0.60f, 0.30f, 0.35f}},
            {"banner_success_text", {0.04f, 0.42f, 0.18f, 1.00f}},
            {"banner_error_bg", {0.85f, 0.20f, 0.20f, 0.08f}},
            {"banner_error_border", {0.85f, 0.20f, 0.20f, 0.35f}},
            {"banner_error_text", {0.60f, 0.12f, 0.12f, 1.00f}},

            // Background Colors
            {"background", {0.97f, 0.97f, 0.98f, 1.00f}},
            {"workspace_particle", {0.149f, 0.149f, 0.149f, 1.0f}},
            {"card", {0.93f, 0.93f, 0.94f, 1.00f}},
            {"agent_composer", {0.95f, 0.95f, 0.96f, 1.00f}},
            {"agent_outline", {0.00f, 0.00f, 0.00f, 0.14f}},
            {"agent_activity", {0.20f, 0.47f, 0.96f, 1.00f}},
            {"popup_bg", {0.96f, 0.96f, 0.97f, 1.0f}},
            {"modal_dim", {0.00f, 0.00f, 0.00f, 0.35f}},
            {"notification_bg", {0.96f, 0.96f, 0.97f, 0.90f}},
            {"pill_field_bg", {0.96f, 0.96f, 0.97f, 0.97f}},
            {"plot_grid", {0.89f, 0.89f, 0.90f, 1.0f}},
            {"plot_grid_major", {0.58f, 0.58f, 0.61f, 1.0f}},
            {"plot_accent", {0.86f, 0.52f, 0.00f, 1.0f}},
            {"plot_halo", {1.00f, 1.00f, 1.00f, 0.75f}},
            {"transparent", {0.00f, 0.00f, 0.00f, 0.00f}},
            {"remote_btn", {0.18f, 0.68f, 0.36f, 1.0f}},
            {"remote_btn_hovered", {0.22f, 0.73f, 0.40f, 1.0f}},

            // Button Colors
            {"action_btn", {0.20f, 0.47f, 0.96f, 1.0f}},
            {"action_btn_hovered", {0.25f, 0.52f, 0.99f, 1.0f}},
            {"action_btn_active", {0.15f, 0.35f, 0.85f, 1.0f}},
            {"action_btn_text", {1.00f, 1.00f, 1.00f, 1.0f}},
            {"action_btn_outline", {0.15f, 0.35f, 0.85f, 0.85f}},
            {"action_btn_highlight", {1.00f, 1.00f, 1.00f, 0.38f}},
            {"destructive_btn", {0.92f, 0.20f, 0.20f, 1.0f}},
            {"destructive_btn_hovered", {0.98f, 0.30f, 0.30f, 1.0f}},
            {"destructive_btn_active", {0.80f, 0.15f, 0.15f, 1.0f}},
            {"destructive_btn_text", {1.00f, 1.00f, 1.00f, 1.0f}},
            {"destructive_btn_outline", {0.74f, 0.12f, 0.12f, 1.0f}},
            {"contrast_btn", {0.12f, 0.12f, 0.14f, 1.0f}},
            {"contrast_btn_hovered", {0.24f, 0.24f, 0.26f, 1.0f}},
            {"contrast_btn_active", {0.34f, 0.34f, 0.36f, 1.0f}},
            {"contrast_btn_text", {1.00f, 1.00f, 1.00f, 1.0f}},
            {"warning_btn", {0.85f, 0.45f, 0.10f, 1.0f}},
            {"warning_btn_hovered", {0.95f, 0.55f, 0.15f, 1.0f}},
            {"warning_btn_active", {0.75f, 0.35f, 0.05f, 1.0f}},
            {"warning_btn_text", {1.00f, 1.00f, 1.00f, 1.0f}},
            {"warning_btn_outline", {0.75f, 0.35f, 0.05f, 0.85f}},

            // Text Colors
            {"text_primary", {0.12f, 0.12f, 0.14f, 1.00f}},
            {"text_secondary", {0.45f, 0.45f, 0.48f, 1.00f}},
            {"text_disabled", {0.65f, 0.65f, 0.68f, 1.0f}},
            {"text_selected_bg", {0.70f, 0.70f, 0.72f, 0.50f}},

            // Editor Colors (One Light inspired)
            {"editor_background", {1.00f, 1.00f, 1.00f, 1.00f}},
            {"editor_text", {0.22f, 0.23f, 0.26f, 1.00f}},
            {"editor_comment", {0.63f, 0.63f, 0.65f, 1.00f}},
            {"editor_keyword", {0.65f, 0.15f, 0.64f, 1.00f}},
            {"editor_string", {0.31f, 0.63f, 0.31f, 1.00f}},
            {"editor_constant", {0.60f, 0.41f, 0.00f, 1.00f}},
            {"editor_function", {0.25f, 0.47f, 0.95f, 1.00f}},
            {"editor_type", {0.76f, 0.52f, 0.00f, 1.00f}},
            {"editor_console_background", {0.95f, 0.95f, 0.96f, 0.96f}},
            {"editor_console_header_outline", {0.82f, 0.82f, 0.85f, 1.00f}},
            {"editor_status_info", {0.93f, 0.96f, 1.00f, 1.00f}},
            {"editor_status_info_text", {0.18f, 0.40f, 0.70f, 1.00f}},
            {"editor_status_info_separator", {0.72f, 0.75f, 0.80f, 0.75f}},
            {"editor_status_success", {0.94f, 0.98f, 0.95f, 1.00f}},
            {"editor_status_success_text", {0.22f, 0.50f, 0.35f, 1.00f}},
            {"editor_status_success_separator", {0.75f, 0.80f, 0.76f, 0.75f}},
            {"editor_status_warning", {1.00f, 0.97f, 0.92f, 1.00f}},
            {"editor_status_warning_text", {0.65f, 0.45f, 0.12f, 1.00f}},
            {"editor_status_warning_separator", {0.80f, 0.77f, 0.70f, 0.75f}},
            {"editor_status_error", {1.00f, 0.94f, 0.94f, 1.00f}},
            {"editor_status_error_text", {0.70f, 0.22f, 0.22f, 1.00f}},
            {"editor_status_error_separator", {0.80f, 0.74f, 0.74f, 0.75f}},
            {"editor_selection", {0.80f, 0.86f, 0.96f, 0.60f}},
            {"editor_selection_match", {1.00f, 0.88f, 0.46f, 0.58f}},
            {"editor_active_line", {0.82f, 0.85f, 0.90f, 0.45f}},
            {"editor_line_number", {0.63f, 0.63f, 0.65f, 1.00f}},
            {"editor_cursor", {0.32f, 0.44f, 1.00f, 1.00f}},
            {"editor_gutter_separator", {0.82f, 0.82f, 0.85f, 0.55f}},
            {"editor_scrollbar_track", {0.87f, 0.87f, 0.88f, 1.00f}},
            {"editor_scrollbar_thumb", {0.55f, 0.55f, 0.57f, 0.80f}},

            // Markdown Callout Colors
            {"callout_note", {0.04f, 0.41f, 0.85f, 1.00f}},
            {"callout_tip", {0.10f, 0.50f, 0.22f, 1.00f}},
            {"callout_important", {0.51f, 0.31f, 0.87f, 1.00f}},
            {"callout_warning", {0.60f, 0.40f, 0.00f, 1.00f}},
            {"callout_caution", {0.81f, 0.13f, 0.18f, 1.00f}},
            {"callout_error", {0.64f, 0.05f, 0.15f, 1.00f}},

            // Markdown Code Block Colors
            {"markdown_code_block", {0.965f, 0.965f, 0.965f, 1.00f}},
            {"markdown_code_block_border", {0.85f, 0.85f, 0.85f, 1.00f}},

            // Chat Colors
            {"chat_user_bubble", {0.20f, 0.47f, 0.96f, 1.00f}},
            {"chat_user_bubble_outline", {0.15f, 0.35f, 0.85f, 0.85f}},

            // Voice Colors
            {"voice_bg", {0.99f, 0.99f, 0.995f, 1.0f}},
            {"voice_listening", {0.18f, 0.46f, 0.92f, 1.0f}},
            {"voice_busy", {0.88f, 0.42f, 0.14f, 1.0f}},
            {"voice_dim", {0.62f, 0.62f, 0.66f, 1.0f}},
            {"voice_stop", {0.82f, 0.18f, 0.16f, 1.0f}},

            // Border Colors
            {"border", {0.78f, 0.78f, 0.80f, 0.75f}},

            // Frame Colors
            {"frame_bg_hovered", {0.86f, 0.86f, 0.87f, 0.95f}},
            {"frame_bg_active", {0.80f, 0.80f, 0.82f, 0.95f}},

            // Scrollbar Colors
            {"scrollbar_bg", {0.0f, 0.0f, 0.0f, 0.0f}},
            {"scrollbar_grab", {0.65f, 0.65f, 0.67f, 0.80f}},

            // Button Colors
            {"button", {0.90f, 0.90f, 0.91f, 1.00f}},
            {"button_hovered", {0.85f, 0.85f, 0.86f, 0.80f}},
            {"button_active", {0.80f, 0.80f, 0.81f, 0.80f}},
            {"button_text", {0.12f, 0.12f, 0.14f, 1.00f}},
            {"button_outline", {0.78f, 0.78f, 0.80f, 0.75f}},

            // Header Colors
            {"header", {0.90f, 0.90f, 0.91f, 0.80f}},
            {"header_hovered", {0.85f, 0.85f, 0.86f, 0.80f}},
            {"header_active", {0.80f, 0.80f, 0.81f, 0.80f}},

            // Separator Colors
            {"separator", {0.80f, 0.80f, 0.82f, 0.50f}},
            {"separator_hovered", {0.70f, 0.70f, 0.72f, 0.80f}},

            // Resize Grip Colors
            {"resize_grip", {0.43f, 0.43f, 0.46f, 0.55f}},
            {"resize_grip_hovered", {0.38f, 0.38f, 0.41f, 0.70f}},
            {"resize_grip_active", {0.32f, 0.32f, 0.35f, 0.85f}},

            // Window Chrome Colors
            {"tile_bg", {0.98f, 0.98f, 0.985f, 1.00f}},
            {"tile_border", {0.00f, 0.00f, 0.00f, 0.10f}},
            {"tile_header_text", {0.32f, 0.32f, 0.35f, 1.00f}},
            {"tile_button_hover", {0.04f, 0.04f, 0.05f, 1.00f}},
            {"tile_button_active_bg", {0.15f, 0.55f, 0.85f, 0.14f}},
            {"tile_shadow", {0.00f, 0.00f, 0.00f, 0.14f}},
            {"drop_fill", {0.15f, 0.45f, 0.95f, 0.14f}},
            {"drop_border", {0.15f, 0.45f, 0.95f, 0.90f}},
            {"pill_bg", {0.92f, 0.92f, 0.93f, 1.00f}},
            {"pill_hover_bg", {0.96f, 0.96f, 0.97f, 1.00f}},
            {"pill_active_bg", {1.00f, 1.00f, 1.00f, 1.00f}},
            {"pill_selected_bg", {0.91f, 0.91f, 0.92f, 1.00f}},
            {"pill_text", {0.14f, 0.14f, 0.16f, 1.00f}},
            {"chrome_button_hover", {0.00f, 0.00f, 0.00f, 0.07f}},

            // Settings Colors
            {"setting_card_border", {0.00f, 0.00f, 0.00f, 0.09f}},
            {"setting_card_hovered", {0.00f, 0.00f, 0.00f, 0.03f}},
            {"primary_btn", {0.10f, 0.10f, 0.10f, 1.00f}},
            {"primary_btn_hovered", {0.22f, 0.22f, 0.22f, 1.00f}},
            {"primary_btn_text", {0.98f, 0.98f, 0.98f, 1.00f}},
            {"toggle_on", {0.20f, 0.45f, 0.95f, 1.00f}},
            {"toggle_on_hovered", {0.16f, 0.40f, 0.90f, 1.00f}},
            {"toggle_off", {0.80f, 0.80f, 0.80f, 1.00f}},
            {"toggle_off_hovered", {0.75f, 0.75f, 0.75f, 1.00f}},
            {"toggle_knob", {1.00f, 1.00f, 1.00f, 1.00f}},

            // Table Colors
            {"table_header_bg", {0.94f, 0.94f, 0.95f, 0.95f}},
            {"table_border_strong", {0.80f, 0.80f, 0.82f, 0.70f}},
            {"table_border_light", {0.88f, 0.88f, 0.90f, 0.50f}},
            {"table_row_bg", {0.00f, 0.00f, 0.00f, 0.00f}},
            {"table_row_bg_alt", {0.00f, 0.00f, 0.00f, 0.02f}},

            // Node Editor Colors
            {"node_background", {1.00f, 1.00f, 1.00f, 1.0f}},
            {"node_outline", {0.92f, 0.62f, 0.08f, 1.0f}},
            {"node_outline_error", {0.86f, 0.24f, 0.24f, 1.0f}},
            {"node_outline_pending", {0.55f, 0.55f, 0.55f, 1.0f}},
            {"node_pin", {0.92f, 0.62f, 0.08f, 1.0f}},
            {"node_link", {0.92f, 0.62f, 0.08f, 1.0f}},
            {"node_toggle_on", {0.40f, 0.80f, 0.40f, 1.0f}},
            {"grid_line", {0.82f, 0.82f, 0.83f, 1.0f}},

        }
    }
};

std::string ResolveThemeKey(const std::string& value) {
    for (const auto& [key, _] : Themes) {
        if (std::equal(key.begin(), key.end(), value.begin(), value.end(),
                       [](unsigned char lhs, unsigned char rhs) {
                           return std::tolower(lhs) == std::tolower(rhs);
                       })) {
            return key;
        }
    }
    return "Dark";
}

std::vector<std::string> BuildThemeKeys() {
    std::vector<std::string> themeKeys;
    themeKeys.reserve(Themes.size());
    for (const auto& [themeKey, _] : Themes) {
        themeKeys.push_back(themeKey);
    }
    std::sort(themeKeys.begin(), themeKeys.end());
    return themeKeys;
}

SurfaceTheme MakeSurfaceTheme(const Palette& palette, SurfacePlacementType placement) {
    SurfaceTheme theme;
    const auto color = [&palette](const char* key, const ColorRGBA<F32>& fallback) {
        const auto it = palette.find(key);
        return it == palette.end() ? fallback : it->second;
    };
    theme.background = color(placement == SurfacePlacementType::Attached ? "node_background" : "tile_bg",
                             theme.background);
    theme.background.a = 1.0f;
    theme.foreground = color("text_primary", theme.foreground);
    theme.secondary = color("text_secondary", theme.secondary);
    theme.grid = color("plot_grid", theme.grid);
    theme.majorGrid = color("plot_grid_major", theme.majorGrid);
    theme.accent = color("plot_accent", theme.accent);
    theme.halo = color("plot_halo", theme.halo);
    theme.pill = color("pill_field_bg", theme.pill);
    return theme;
}

}  // namespace Jetstream::Sakura
