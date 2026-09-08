#include "jetstream/platform.hh"

#if defined(JST_OS_BROWSER)
#include <GLFW/emscripten_glfw3.h>
#elif defined(JST_OS_LINUX)
#include <cerrno>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#elif defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream::Platform {

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result OpenUrl(const std::string& url) {
    emscripten::glfw3::OpenURL(url, "_blank");
    return Result::SUCCESS;
}

Result OpenFolder(const std::string& path) {
    (void)path;
    JST_ERROR("Opening folders is not supported on this platform.");
    return Result::ERROR;
}

#elif defined(JST_OS_LINUX)

namespace {

[[noreturn]] void ExitXdgLauncherWithError(int descriptor) {
    const int error = errno == 0 ? EIO : errno;
    ssize_t written = 0;
    do {
        written = write(descriptor, &error, sizeof(error));
    } while (written < 0 && errno == EINTR);
    _exit(127);
}

Result OpenWithXdg(const std::string& target) {
    int errorPipe[2];
    if (pipe(errorPipe) != 0) {
        JST_ERROR("Failed to create xdg-open launcher pipe.");
        return Result::ERROR;
    }

    if (errorPipe[1] <= STDERR_FILENO) {
        const int relocated = fcntl(
            errorPipe[1], F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
        close(errorPipe[1]);
        errorPipe[1] = relocated;
    } else {
        const int flags = fcntl(errorPipe[1], F_GETFD, 0);
        if (flags < 0 ||
            fcntl(errorPipe[1], F_SETFD, flags | FD_CLOEXEC) < 0) {
            close(errorPipe[0]);
            close(errorPipe[1]);
            JST_ERROR("Failed to prepare xdg-open launcher pipe.");
            return Result::ERROR;
        }
    }
    if (errorPipe[1] < 0) {
        close(errorPipe[0]);
        JST_ERROR("Failed to prepare xdg-open launcher pipe.");
        return Result::ERROR;
    }

    const pid_t intermediate = fork();
    if (intermediate < 0) {
        close(errorPipe[0]);
        close(errorPipe[1]);
        JST_ERROR("Failed to launch xdg-open.");
        return Result::ERROR;
    }

    if (intermediate == 0) {
        close(errorPipe[0]);
        const pid_t launcher = fork();
        if (launcher < 0) {
            ExitXdgLauncherWithError(errorPipe[1]);
        }
        if (launcher > 0) {
            close(errorPipe[1]);
            _exit(0);
        }

        if (setsid() < 0) {
            ExitXdgLauncherWithError(errorPipe[1]);
        }
        const int nullDevice = open("/dev/null", O_RDWR);
        if (nullDevice < 0 ||
            dup2(nullDevice, STDIN_FILENO) < 0 ||
            dup2(nullDevice, STDOUT_FILENO) < 0 ||
            dup2(nullDevice, STDERR_FILENO) < 0) {
            ExitXdgLauncherWithError(errorPipe[1]);
        }
        if (nullDevice > STDERR_FILENO) {
            close(nullDevice);
        }

        execlp("xdg-open",
               "xdg-open",
               target.c_str(),
               static_cast<char*>(nullptr));
        ExitXdgLauncherWithError(errorPipe[1]);
    }

    close(errorPipe[1]);
    int status = 0;
    pid_t result = 0;
    do {
        result = waitpid(intermediate, &status, 0);
    } while (result < 0 && errno == EINTR);

    int launchError = 0;
    ssize_t bytesRead = 0;
    do {
        bytesRead = read(errorPipe[0], &launchError, sizeof(launchError));
    } while (bytesRead < 0 && errno == EINTR);
    close(errorPipe[0]);

    if (result < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        bytesRead != 0) {
        JST_ERROR("Failed to launch xdg-open.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

}  // namespace

Result OpenUrl(const std::string& url) {
    return OpenWithXdg(url);
}

Result OpenFolder(const std::string& path) {
    return OpenWithXdg(path);
}

#elif defined(JST_OS_WINDOWS)

Result OpenUrl(const std::string& url) {
    const INT_PTR result = reinterpret_cast<INT_PTR>(
        ShellExecuteA(nullptr, nullptr, url.c_str(), nullptr, nullptr, SW_SHOW));
    if (result <= 32) {
        JST_ERROR("Failed to open URL: {}", url);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result OpenFolder(const std::string& path) {
    const auto nativePath = PathFromUtf8(path);
    const INT_PTR result = reinterpret_cast<INT_PTR>(
        ShellExecuteW(nullptr,
                      L"open",
                      nativePath.c_str(),
                      nullptr,
                      nullptr,
                      SW_SHOW));
    if (result <= 32) {
        JST_ERROR("Failed to open folder '{}' [Error: {}].", path, result);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

#else

Result OpenUrl(const std::string& url) {
    JST_ERROR("Opening URL is not supported in this platform.");
    return Result::ERROR;
}

Result OpenFolder(const std::string& path) {
    (void)path;
    JST_ERROR("Opening folders is not supported on this platform.");
    return Result::ERROR;
}

#endif

}  // namespace Jetstream::Platform
