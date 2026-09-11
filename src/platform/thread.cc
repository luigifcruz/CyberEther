#include "jetstream/platform.hh"

#include <algorithm>
#include <cerrno>
#include <exception>
#include <system_error>
#include <utility>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#undef ERROR
#undef FATAL
#elif defined(JST_OS_BROWSER)
#include <thread>
#else
#include <pthread.h>
#endif

namespace Jetstream::Platform {

struct WorkerThread::Impl {
    std::function<void()> fn;

#if defined(JST_OS_WINDOWS)
    HANDLE handle = nullptr;
#elif defined(JST_OS_BROWSER)
    std::thread handle;
#else
    pthread_t handle{};
#endif

#if defined(JST_OS_WINDOWS)
    static unsigned int __stdcall run(void* context) noexcept {
#else
    static void* run(void* context) noexcept {
#endif
        std::function<void()> fn;
        fn.swap(static_cast<Impl*>(context)->fn);
        fn();
        return {};
    }
};

WorkerThread::WorkerThread() = default;

WorkerThread::~WorkerThread() {
    if (joinable()) {
        std::terminate();
    }
}

void WorkerThread::start(std::function<void()> fn) {
    if (joinable()) {
        std::terminate();
    }

    auto state = std::make_unique<Impl>();
    state->fn.swap(fn);

#if defined(JST_OS_BROWSER)
    state->handle = std::thread(Impl::run, state.get());
#else
    constexpr unsigned int StackSize = 8 * 1024 * 1024;
#if defined(JST_OS_WINDOWS)
    state->handle = reinterpret_cast<HANDLE>(_beginthreadex(
        nullptr, StackSize, Impl::run, state.get(),
        STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr));
    if (!state->handle) {
        throw std::system_error(errno, std::generic_category(), "_beginthreadex");
    }
#else
    pthread_attr_t attributes;
    int error = pthread_attr_init(&attributes);
    if (error != 0) {
        throw std::system_error(error, std::generic_category(), "pthread_attr_init");
    }

    std::size_t stackSize = 0;
    error = pthread_attr_getstacksize(&attributes, &stackSize);
    if (error == 0) {
        error = pthread_attr_setstacksize(&attributes,
                                          std::max<std::size_t>(stackSize, StackSize));
    }
    if (error == 0) {
        error = pthread_create(&state->handle, &attributes, Impl::run, state.get());
    }
    pthread_attr_destroy(&attributes);
    if (error != 0) {
        throw std::system_error(error, std::generic_category(), "Worker thread creation");
    }
#endif
#endif

    impl = std::move(state);
}

void WorkerThread::join() {
    if (!joinable()) {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument));
    }

#if defined(JST_OS_WINDOWS)
    if (GetThreadId(impl->handle) == GetCurrentThreadId()) {
        throw std::system_error(std::make_error_code(std::errc::resource_deadlock_would_occur));
    }
    if (WaitForSingleObject(impl->handle, INFINITE) != WAIT_OBJECT_0) {
        throw std::system_error(GetLastError(), std::system_category(), "Worker thread join");
    }
    CloseHandle(impl->handle);
#elif defined(JST_OS_BROWSER)
    impl->handle.join();
#else
    const int error = pthread_join(impl->handle, nullptr);
    if (error != 0) {
        throw std::system_error(error, std::generic_category(), "pthread_join");
    }
#endif

    impl.reset();
}

bool WorkerThread::joinable() const {
    return impl != nullptr;
}

}  // namespace Jetstream::Platform
