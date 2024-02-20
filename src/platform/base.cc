#include "jetstream/platform.hh"

#if defined(JST_OS_IOS) || defined(JST_OS_MAC)
#include "apple.hh"
#endif

#if defined(JST_OS_BROWSER)
#include "emscripten.h"

EM_JS(const char*, jst_file_error_string, (), {
    return stringToNewUTF8(window.jst.error_string);
});

EM_JS(int, jst_file_error, (), {
    return window.jst.error;
});

EM_JS(const char*, jst_file_path, (), {
    return stringToNewUTF8(window.jst.path);
});
#endif

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <comutil.h>
#include <commdlg.h>
#include <stdio.h>
#include <shellapi.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream::Platform {

//
// Open URL
//

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result OpenUrl(const std::string& url) {
    emscripten_run_script(jst::fmt::format("window.open('{}', '_blank');", url).c_str());
    return Result::SUCCESS;
}

#elif defined(JST_OS_LINUX)

Result OpenUrl(const std::string& url) {
    const auto res = system(jst::fmt::format("xdg-open ""{}""", url).c_str());
    if (res != 0) {
        JST_ERROR("Failed to open URL: {}", url);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)
Result OpenUrl(const std::string& url) {
    INT_PTR res = (INT_PTR)ShellExecuteA(0, 0, url.c_str(), 0, 0 , SW_SHOW );
    if (res < 32) {
        JST_ERROR("Failed to open URL: {}", url);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

#else

Result OpenUrl(const std::string& url) {
    JST_ERROR("Opening URL is not supported in this platform.");
    return Result::ERROR;
}

#endif

//
// Pick File
//

// TODO: Implement iOS support.

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

EM_ASYNC_JS(void, jst_pick_file, (), {
    if (typeof window === 'undefined') {
        console.error("File Picker: window is not defined.");
        return;
    }

    if (!window.showOpenFilePicker) {
        window.jst.error = 1;
        window.jst.error_string = "File picker is not supported in this browser.";
        return;
    }

    return new Promise((resolve) => {
        const options = {
            types: [
                {
                    description: 'Flowgraph Files',
                    accept: {
                        'text/plain': ['.yml', '.yaml'],
                    },
                },
            ],
            multiple: false,
        };

        window
            .showOpenFilePicker(options)
            .then(async ([handle]) => {
                FS.createPath("/", "vfs");

                const filepath = "/vfs/" + handle.name;
                if (FS.analyzePath(filepath).exists) {
                    FS.unlink(filepath);
                }

                const file = await handle.getFile();
                const arrayBuffer = await file.arrayBuffer();
                FS.createDataFile("/vfs/", handle.name, new Uint8Array(arrayBuffer), true, true);

                window.jst.fsHandle = handle;
                window.jst.error = 0;
                window.jst.path = filepath;
                resolve();
            })
            .catch(err => {
                window.jst.error = 1;
                window.jst.error_string = "User cancelled file picker.";
                resolve();
            });
    });
});

Result PickFile(std::string& path) {
    jst_pick_file();
    if (jst_file_error()) {
        JST_ERROR("{}", jst_file_error_string());
        return Result::ERROR;
    }
    path = std::string(jst_file_path());
    return Result::SUCCESS;
}

#elif defined(JST_OS_LINUX)

Result PickFile(std::string& path) {
    std::array<char, 1024> buffer;
    std::string command = "zenity --file-selection --file-filter='YAML files | *.yml *.yaml' 2>/dev/null";

    auto pipe_deleter = [](FILE* file) { if (file) pclose(file); };
    std::unique_ptr<FILE, decltype(pipe_deleter)> pipe(popen(command.c_str(), "r"), pipe_deleter);

    if (!pipe) {
        JST_ERROR("Failed to open file selection dialog.");
        return Result::ERROR;
    }

    if (pipe.get() == nullptr) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    const auto res = fgets(buffer.data(), buffer.size(), pipe.get());
    if (res == nullptr) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = buffer.data();

    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    if (path.back() == '\n') {
        path.pop_back();
    }

    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

Result PickFile(std::string& path) {
    /* File path buffer */
    char buf[256] = {'\0'};
    memcpy_s(buf,256,path.c_str(),path.length());

    /* Create OpenFilenameA Struct */
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(OPENFILENAME));

    /* Fill struct */
    ofn.lStructSize = sizeof(OPENFILENAME);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = buf;
    ofn.nMaxFile = 256;
    ofn.lpstrFilter = "All Files\0*.*\0CyberEther Flowgraphs (.yml, .yaml)\0*.yml;*.yaml\0";
    ofn.nFilterIndex = 2;

    bool ret = GetOpenFileNameA(&ofn);
    if (!ret) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = std::string(ofn.lpstrFile);

    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    if (path.back() == '\n') {
        path.pop_back();
    }

    return Result::SUCCESS;
}

#else

Result PickFile(std::string& path) {
    JST_ERROR("Picking files is not supported in this platform.");
    return Result::ERROR;
}

#endif

//
// Save File
//

// TODO: Implement iOS support.

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

EM_ASYNC_JS(void, jst_save_file, (), {
    if (typeof window === 'undefined') {
        console.error("Save File Picker: window is not defined.");
        return;
    }

    if (!window.showSaveFilePicker) {
        window.jst.error = 1;
        window.jst.error_string = "Save file picker is not supported in this browser.";
        return;
    }

    return new Promise((resolve) => {
        const options = {
            types: [
                {
                    description: 'Flowgraph Files',
                    accept: {
                        'text/plain': ['.yml', '.yaml'],
                    },
                },
            ],
        };

        window
            .showSaveFilePicker(options)
            .then(async (handle) => {
                FS.createPath("/", "vfs");

                const filepath = "/vfs/" + handle.name;
                if (FS.analyzePath(filepath).exists) {
                    FS.unlink(filepath);
                }

                FS.createDataFile("/vfs/", handle.name, new Uint8Array(), true, true);

                window.jst.fsHandle = handle;
                window.jst.error = 0;
                window.jst.path = filepath;
                resolve();
            })
            .catch(err => {
                window.jst.error = 1;
                window.jst.error_string = "User cancelled save file picker.";
                resolve();
            });
    });
});

Result SaveFile(std::string& path) {
    jst_save_file();
    if (jst_file_error()) {
        JST_ERROR("{}", jst_file_error_string());
        return Result::ERROR;
    }
    path = std::string(jst_file_path());
    return Result::SUCCESS;
}

#elif defined(JST_OS_LINUX)

Result SaveFile(std::string& path) {
    std::array<char, 1024> buffer;
    std::string command = "zenity --file-selection --save --confirm-overwrite --file-filter='YAML files | *.yml *.yaml' 2>/dev/null";

    auto pipe_deleter = [](FILE* file) { if (file) pclose(file); };
    std::unique_ptr<FILE, decltype(pipe_deleter)> pipe(popen(command.c_str(), "r"), pipe_deleter);

    if (!pipe) {
        JST_ERROR("Failed to open save file dialog.");
        return Result::ERROR;
    }

    if (pipe.get() == nullptr) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    const auto res = fgets(buffer.data(), buffer.size(), pipe.get());
    if (res == nullptr) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = buffer.data();

    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    if (path.back() == '\n') {
        path.pop_back();
    }

    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

Result SaveFile(std::string& path) {
    /* File path buffer */
    char buf[256] = {'\0'};
    memcpy_s(buf,256,path.c_str(),path.length());

    /* Create OpenFilenameA Struct */
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(OPENFILENAME));

    /* Fill struct */
    ofn.lStructSize = sizeof(OPENFILENAME);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = buf;
    ofn.nMaxFile = 256;
    ofn.lpstrFilter = "All Files\0*.*\0CyberEther Flowgraphs (.yml, .yaml)\0*.yml;*.yaml\0";
    ofn.nFilterIndex = 2;

    bool ret = GetSaveFileNameA(&ofn);
    if (!ret) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = std::string(ofn.lpstrFile);

    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    if (path.back() == '\n') {
        path.pop_back();
    }

    return Result::SUCCESS;
}

#else

Result SaveFile(std::string& path) {
    JST_ERROR("Saving files is not supported in this platform.");
    return Result::ERROR;
}

#endif

}  // namespace Jetstream::Platform