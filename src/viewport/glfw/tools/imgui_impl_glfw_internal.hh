#ifndef JETSTREAM_IMGUI_IMPL_GLFW_INTERNAL_HH
#define JETSTREAM_IMGUI_IMPL_GLFW_INTERNAL_HH

#include <functional>

struct GLFWwindow;

std::function<void()> ImGui_ImplGlfw_MakeLeftButtonRelease(GLFWwindow* window);

#endif
