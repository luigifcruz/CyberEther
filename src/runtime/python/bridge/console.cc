#include <cstddef>
#include <utility>

#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/cpython/base.hh"

namespace Jetstream {

using namespace CPython;

namespace {

constexpr std::size_t kMaxConsoleLines = 256;

void TrimTrailingCarriageReturns(std::string& line) {
    while (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

void TrimToMaxLines(std::vector<std::string>& lines) {
    const auto overflow = lines.size() > kMaxConsoleLines ? lines.size() - kMaxConsoleLines : 0;
    if (overflow > 0) {
        lines.erase(lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(overflow));
    }
}

void AppendLine(std::vector<std::string>& lines, std::string line) {
    TrimTrailingCarriageReturns(line);
    lines.push_back(std::move(line));
    TrimToMaxLines(lines);
}

void DiscardPythonError() {
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);

    if (type) { Py_DecRef(type); }
    if (value) { Py_DecRef(value); }
    if (traceback) { Py_DecRef(traceback); }
}

std::string PythonToString(PyObject* object) {
    if (!object) {
        return {};
    }

    auto* str = PyObject_Str(object);
    if (!str) {
        DiscardPythonError();
        return {};
    }

    const char* chars = PyUnicode_AsUTF8(str);
    std::string result = chars ? chars : "";
    Py_DecRef(str);

    return result;
}

std::vector<std::string> PythonToStrings(PyObject* sequence) {
    std::vector<std::string> lines;

    if (!sequence) {
        return lines;
    }

    const auto size = PySequence_Size(sequence);
    if (size < 0) {
        DiscardPythonError();
        return lines;
    }

    lines.reserve(static_cast<std::size_t>(size));
    for (Py_ssize_t i = 0; i < size; ++i) {
        auto* item = PySequence_GetItem(sequence, i);
        if (!item) {
            DiscardPythonError();
            continue;
        }

        lines.push_back(PythonToString(item));
        Py_DecRef(item);
    }

    return lines;
}

}  // namespace

void Bridge::consoleClear() {
    std::lock_guard<std::mutex> lock(consoleMutex);
    consoleLines.clear();
}

void Bridge::consoleAppend(const std::string& text) {
    if (text.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(consoleMutex);

    std::size_t start = 0;
    while (start < text.size()) {
        const auto newline = text.find('\n', start);
        if (newline == std::string::npos) {
            AppendLine(consoleLines, text.substr(start));
            return;
        }

        AppendLine(consoleLines, text.substr(start, newline - start));
        start = newline + 1;
    }
}

bool Bridge::consoleRefresh() {
    if (!globals) {
        return false;
    }

    auto* snapshotFunction = PyDict_GetItemString(globals, "_jetstream_console_snapshot");
    if (!snapshotFunction || !PyCallable_Check(snapshotFunction)) {
        return false;
    }

    auto* output = PyObject_CallFunctionObjArgs(snapshotFunction);
    if (!output) {
        DiscardPythonError();
        return false;
    }

    auto capturedLines = PythonToStrings(output);
    Py_DecRef(output);
    TrimToMaxLines(capturedLines);

    std::lock_guard<std::mutex> lock(consoleMutex);
    consoleLines = std::move(capturedLines);
    return true;
}

}  // namespace Jetstream
