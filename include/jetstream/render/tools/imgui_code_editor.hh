// imgui_code_editor: Multiline text editor with line numbers.
// Reusable ImGui widget for editing text with a gutter showing
// line numbers, similar to a basic code editor.
//
// Usage:
//   std::string text = "Hello\nWorld";
//   bool changed = ImGui::InputTextCodeEditor("##id", &text, ImVec2(0, 300));
//
// Requires: imgui.h, imgui_internal.h, imgui_stdlib.h

#pragma once

#ifndef IMGUI_DISABLE

#include <string>
#include <cstdio>

namespace ImGui {

inline bool InputTextCodeEditor(
        const char* label,
        std::string* str,
        const ImVec2& size = ImVec2(0, 0),
        ImGuiInputTextFlags flags = 0,
        ImGuiInputTextCallback callback = nullptr,
        void* user_data = nullptr) {
    // Count lines.
    int lineCount = 1;
    for (const char c : *str) {
        if (c == '\n') {
            lineCount++;
        }
    }

    // Measure gutter width based on the number of digits.
    char buf[16];
    snprintf(buf, sizeof(buf), " %d ", lineCount);
    const float gutterWidth = CalcTextSize(buf).x;
    const float lineHeight = GetTextLineHeight();

    // Resolve total size (0 means fill available).
    ImVec2 totalSize = size;
    if (totalSize.x <= 0.0f) {
        totalSize.x = GetContentRegionAvail().x + totalSize.x;
    }
    if (totalSize.y <= 0.0f) {
        totalSize.y = GetContentRegionAvail().y + totalSize.y;
    }

    const float editorWidth = totalSize.x - gutterWidth;

    PushID(label);

    // Compute the child window ID that InputTextMultiline will
    // create. It calls BeginChildEx with the label and the ID
    // derived from it in the current ID stack.
    const ImGuiID editorChildId = GetCurrentWindow()->GetID("##editor");

    // Render the multiline input offset to the right for the gutter.
    SetCursorPosX(GetCursorPosX() + gutterWidth);

    bool changed = InputTextMultiline(
        "##editor", str,
        ImVec2(editorWidth, totalSize.y),
        flags | ImGuiInputTextFlags_AllowTabInput,
        callback, user_data);

    // Grab the scroll position from the InputTextMultiline child
    // window.
    float scrollY = 0.0f;
    if (ImGuiWindow* editorWindow = FindWindowByID(editorChildId)) {
        scrollY = editorWindow->Scroll.y;
    }

    // Position the gutter to the left of the editor, same row.
    const ImVec2 editorMin = GetItemRectMin();
    const ImVec2 editorMax = GetItemRectMax();

    // Draw gutter background.
    ImDrawList* drawList = GetWindowDrawList();
    const ImVec2 gutterMin(editorMin.x - gutterWidth, editorMin.y);
    const ImVec2 gutterMax(editorMin.x, editorMax.y);

    const ImU32 gutterBg = GetColorU32(ImGuiCol_FrameBg, 0.6f);
    const ImU32 gutterText = GetColorU32(ImGuiCol_TextDisabled);
    const ImU32 gutterBorder = GetColorU32(ImGuiCol_Border);
    const float rounding = GetStyle().FrameRounding;

    drawList->AddRectFilled(gutterMin, gutterMax, gutterBg,
                            rounding, ImDrawFlags_RoundCornersLeft);

    // Overdraw the editor's left rounded corners so they appear
    // flat where the gutter meets the text area.
    const ImU32 editorBg = GetColorU32(ImGuiCol_FrameBg);
    drawList->AddRectFilled(
        ImVec2(gutterMax.x, gutterMin.y),
        ImVec2(gutterMax.x + rounding, gutterMin.y + rounding),
        editorBg);
    drawList->AddRectFilled(
        ImVec2(gutterMax.x, gutterMax.y - rounding),
        ImVec2(gutterMax.x + rounding, gutterMax.y),
        editorBg);

    drawList->AddLine(ImVec2(gutterMax.x, gutterMin.y),
                      gutterMax, gutterBorder);

    // Clip line number rendering to the gutter area.
    drawList->PushClipRect(gutterMin, gutterMax, true);

    // The editor content has FramePadding at top.
    const float padY = GetStyle().FramePadding.y;

    // Draw only visible line numbers.
    const int firstVisible = static_cast<int>(scrollY / lineHeight);
    const int visibleLines = static_cast<int>(
        totalSize.y / lineHeight) + 2;

    for (int i = firstVisible;
         i < lineCount && i < firstVisible + visibleLines; i++) {
        const float y = gutterMin.y + padY
                        + (i * lineHeight) - scrollY;

        if (y > gutterMax.y) break;
        if (y + lineHeight < gutterMin.y) continue;

        snprintf(buf, sizeof(buf), "%d", i + 1);
        const float textWidth = CalcTextSize(buf).x;
        // Right-align within gutter with small right padding.
        const float x = gutterMax.x - textWidth - 4.0f;
        drawList->AddText(ImVec2(x, y), gutterText, buf);
    }

    drawList->PopClipRect();

    PopID();

    return changed;
}

}  // namespace ImGui

#endif  // IMGUI_DISABLE
