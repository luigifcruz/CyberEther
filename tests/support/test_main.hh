#ifndef JETSTREAM_TESTS_SUPPORT_TEST_MAIN_HH
#define JETSTREAM_TESTS_SUPPORT_TEST_MAIN_HH

#include <catch2/catch_session.hpp>

#include "jetstream/backend/base.hh"

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#include <string>
#include <vector>
#endif

namespace TestSupport {

inline int Run(int argc, char* argv[]) {
#ifdef JST_OS_BROWSER
    std::vector<std::string> arguments(argv, argv + argc);
    const auto result = Jetstream::Backend::WebGPU::InitializeAsync(
        [arguments = std::move(arguments)](Jetstream::Result status) mutable {
            if (status != Jetstream::Result::SUCCESS) {
                jst::fmt::print("[WASM] WebGPU adapter unavailable; running CPU-compatible cases.\n");
            }
            std::vector<char*> pointers;
            for (auto& argument : arguments) pointers.push_back(argument.data());
            const int result = Catch::Session().run(static_cast<int>(pointers.size()), pointers.data());
            Jetstream::Backend::DestroyAll();
            Jetstream::Backend::WebGPU::CancelInitialization();
            emscripten_force_exit(result);
        });
    if (result != Jetstream::Result::SUCCESS) return 1;
    emscripten_exit_with_live_runtime();
#else
    const int result = Catch::Session().run(argc, argv);
    Jetstream::Backend::DestroyAll();
    return result;
#endif
}

}

#endif
