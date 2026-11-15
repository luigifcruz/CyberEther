#include <cstdint>

#if defined(_WIN32)
#define CRASH_MODULE_EXPORT __declspec(dllexport)
#define CRASH_MODULE_NOINLINE __declspec(noinline)
#else
#define CRASH_MODULE_EXPORT __attribute__((visibility("default")))
#define CRASH_MODULE_NOINLINE __attribute__((noinline))
#endif

namespace {

volatile std::uint32_t completed = 0;

CRASH_MODULE_NOINLINE void CrashModuleFault(volatile std::uint32_t* address) {
    *address = 42;
}

}  // namespace

extern "C" CRASH_MODULE_EXPORT CRASH_MODULE_NOINLINE void CrashModuleCompute() {
    CrashModuleFault(reinterpret_cast<volatile std::uint32_t*>(1));
    completed = completed + 1;
}
