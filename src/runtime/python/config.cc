#include <jetstream/runtime_context_python.hh>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <optional>
#include <ranges>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace Jetstream {

namespace {

std::string Lowercase(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

#if defined(_WIN32)
std::string CompactPythonVersion(const std::string& version) {
    std::string compact;
    compact.reserve(version.size());
    std::ranges::copy_if(version, std::back_inserter(compact), [](unsigned char c) {
        return std::isdigit(c);
    });
    return compact;
}
#endif

#if defined(_WIN32)
#define JST_POPEN _popen
#define JST_PCLOSE _pclose
#else
#define JST_POPEN popen
#define JST_PCLOSE pclose
#endif

std::string Trim(const std::string& value) {
    const auto isSpace = [](unsigned char c) { return std::isspace(c); };
    const auto begin = std::ranges::find_if_not(value, isSpace);
    const auto end = std::ranges::find_if_not(value.rbegin(), value.rend(), isSpace).base();

    if (begin >= end) {
        return {};
    }

    return std::string(begin, end);
}

std::string LastNonEmptyLine(const std::string& value) {
    std::string result;
    std::size_t start = 0;
    while (start < value.size()) {
        const auto end = value.find('\n', start);
        const auto line = Trim(value.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!line.empty()) {
            result = line;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return result;
}

std::string ShellQuote(const std::string& value) {
#if defined(_WIN32)
    std::string quoted = "\"";
    for (const char c : value) {
        if (c == '"') {
            quoted += "\\\"";
        } else {
            quoted += c;
        }
    }
    quoted += "\"";
    return quoted;
#else
    std::string quoted = "'";
    for (const char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
#endif
}

std::filesystem::path ExpandUserPath(const std::string& value) {
    if (value == "~" || value.starts_with("~/") || value.starts_with("~\\")) {
        const char* home = std::getenv("HOME");
#if defined(_WIN32)
        if (!home) {
            home = std::getenv("USERPROFILE");
        }
#endif
        if (home) {
            if (value == "~") {
                return std::filesystem::path(home);
            }
            return std::filesystem::path(home) / value.substr(2);
        }
    }
    return std::filesystem::path(value);
}

void AddPath(std::vector<std::string>& paths, const std::filesystem::path& path) {
    if (path.empty()) {
        return;
    }

    const auto value = path.string();
    if (value.empty() || std::ranges::find(paths, value) != paths.end()) {
        return;
    }

    paths.push_back(value);
}

std::optional<std::string> ExtractPythonVersion(const std::filesystem::path& path) {
    static const std::regex versionRegex(R"((\d+)\.(\d+)(?:\.\d+)?)");

    for (auto current = path; !current.empty();) {
        const auto name = current.filename().string();
        std::smatch match;
        if (std::regex_search(name, match, versionRegex)) {
            return match[1].str() + "." + match[2].str();
        }

        const auto parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }

    return std::nullopt;
}

bool LooksLikeFrameworkLibrary(const std::filesystem::path& path) {
    const auto leaf = Lowercase(path.filename().string());
    for (auto current = path.parent_path(); !current.empty();) {
        const auto name = Lowercase(current.filename().string());
        constexpr std::string_view suffix = ".framework";
        if (name.ends_with(suffix)) {
            const auto frameworkName = name.substr(0, name.size() - suffix.size());
            return leaf == frameworkName;
        }

        const auto parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }

    return false;
}

bool LooksLikePythonExecutable(const std::string& value) {
    const auto path = std::filesystem::path(value);
    if (LooksLikeFrameworkLibrary(path)) {
        return false;
    }

    auto name = Lowercase(path.filename().string());
#if defined(_WIN32)
    if (name == "py.exe") {
        return true;
    }
    if (name.ends_with(".exe")) {
        name.resize(name.size() - std::string_view(".exe").size());
    }
#endif

    if (name == "python" || name == "python3") {
        return true;
    }

    constexpr std::string_view prefix = "python3.";
    if (!name.starts_with(prefix)) {
        return false;
    }

    return std::ranges::all_of(std::string_view(name).substr(prefix.size()), [](unsigned char c) {
        return std::isdigit(c) || c == '.';
    });
}

std::optional<std::string> RunPythonProbe(const std::string& pythonPath, const char* script) {
    if (pythonPath.empty()) {
        return std::nullopt;
    }

    std::string command = ShellQuote(pythonPath) + " -c " + ShellQuote(script);
#if defined(_WIN32)
    command += " 2>NUL";
#else
    command += " 2>/dev/null";
#endif

    FILE* pipe = JST_POPEN(command.c_str(), "r");
    if (!pipe) {
        return std::nullopt;
    }

    std::string output;
    char buffer[1024];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    (void)JST_PCLOSE(pipe);

    const auto value = LastNonEmptyLine(output);
    if (value.empty()) {
        return std::nullopt;
    }

    return value;
}

std::optional<std::string> ProbePythonLibraryPath(const std::string& pythonPath) {
    constexpr const char* kProbeScript =
        "import ctypes.util, os, sys, sysconfig; "
        "c=[]; "
        "v='{}{}'.format(sys.version_info.major, sys.version_info.minor); "
        "n='python{}.dll'.format(v); "
        "[c.append(os.path.join(d, n)) for d in "
        "(os.path.dirname(sys.executable), sys.base_prefix, sys.prefix, "
        "sysconfig.get_config_var('BINDIR'), sysconfig.get_config_var('installed_base')) if d]; "
        "[c.append(os.path.join(d, os.path.basename(l))) "
        "for d in (sys.base_prefix, sys.prefix, sysconfig.get_config_var('installed_base')) "
        "for l in (sysconfig.get_config_var('LDLIBRARY'), sysconfig.get_config_var('INSTSONAME')) "
        "if d and l]; "
        "[c.append(os.path.join(sysconfig.get_config_var(d), sysconfig.get_config_var(l))) "
        "for d in ('LIBDIR','LIBPL') for l in ('LDLIBRARY','INSTSONAME') "
        "if sysconfig.get_config_var(d) and sysconfig.get_config_var(l)]; "
        "f=ctypes.util.find_library('python{}'.format(v)); "
        "c.extend([f] if f else []); "
        "print(next((p for p in c if p and os.path.exists(p)), ''))";

    return RunPythonProbe(pythonPath, kProbeScript);
}

std::optional<std::string> ProbePythonVersion(const std::string& pythonPath) {
    constexpr const char* kProbeScript =
        "import sys; "
        "print('Python {}.{}.{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))";

    return RunPythonProbe(pythonPath, kProbeScript);
}

std::vector<std::string> PythonLibraryCandidates(const std::string& configuredPath) {
    std::vector<std::string> candidates;
    const auto expandedPath = ExpandUserPath(configuredPath);
    const bool isPythonExecutable = LooksLikePythonExecutable(expandedPath.string());

    if (!isPythonExecutable) {
        AddPath(candidates, expandedPath);
    }

    std::error_code ec;
    const auto canonicalPath = std::filesystem::weakly_canonical(expandedPath, ec);
    if (!ec && !LooksLikePythonExecutable(canonicalPath.string())) {
        AddPath(candidates, canonicalPath);
    }

    if (isPythonExecutable) {
        if (const auto probedPath = ProbePythonLibraryPath(expandedPath.string())) {
            AddPath(candidates, *probedPath);
        }
    }

    auto addDerivedCandidates = [&](const std::filesystem::path& path) {
        const auto version = ExtractPythonVersion(path);
        const auto parent = path.parent_path();
        const auto parentName = Lowercase(parent.filename().string());
        const auto root = parentName == "bin" || parentName == "scripts" ? parent.parent_path() : parent;

        if (parentName == "bin") {
            AddPath(candidates, parent.parent_path() / "Python");
        }

        if (!version.has_value()) {
            return;
        }

#if defined(_WIN32)
        const auto compactVersion = CompactPythonVersion(*version);
        if (!compactVersion.empty()) {
            AddPath(candidates, parent / ("python" + compactVersion + ".dll"));
            AddPath(candidates, root / ("python" + compactVersion + ".dll"));
        }
#else
        AddPath(candidates, root / "Frameworks" / "Python.framework" / "Versions" / *version / "Python");
        AddPath(candidates, root / "lib" / ("libpython" + *version + ".dylib"));
        AddPath(candidates, root / "lib" / ("libpython" + *version + ".so"));
        AddPath(candidates, root / "lib" / ("libpython" + *version + ".so.1.0"));
#endif
    };

    if (isPythonExecutable) {
        addDerivedCandidates(expandedPath);
        if (!ec) {
            addDerivedCandidates(canonicalPath);
        }
    }

    return candidates;
}

bool FileIsRegular(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec) && !ec;
}

void AddExecutableFromPath(std::vector<std::string>& paths, const std::filesystem::path& path) {
    if (!LooksLikePythonExecutable(path.string()) || !FileIsRegular(path)) {
        return;
    }

    AddPath(paths, path);
}

void AddExecutablesFromDirectory(std::vector<std::string>& paths, const std::filesystem::path& directory) {
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec) || ec) {
        return;
    }

#if defined(_WIN32)
    AddExecutableFromPath(paths, directory / "py.exe");
    AddExecutableFromPath(paths, directory / "python.exe");
    AddExecutableFromPath(paths, directory / "python3.exe");
#else
    AddExecutableFromPath(paths, directory / "python");
    AddExecutableFromPath(paths, directory / "python3");
#endif

    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
        if (ec) {
            break;
        }
        AddExecutableFromPath(paths, entry.path());
    }
}

void AddPathEnvironmentExecutables(std::vector<std::string>& paths) {
    const char* rawPath = std::getenv("PATH");
    if (!rawPath) {
        return;
    }

    std::string path(rawPath);
#if defined(_WIN32)
    constexpr char kSeparator = ';';
#else
    constexpr char kSeparator = ':';
#endif

    std::size_t start = 0;
    while (start <= path.size()) {
        const auto end = path.find(kSeparator, start);
        const auto directory = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!directory.empty()) {
            AddExecutablesFromDirectory(paths, directory);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

void AddCommonPythonExecutables(std::vector<std::string>& paths) {
    AddPathEnvironmentExecutables(paths);

    for (const char* variable : {"VIRTUAL_ENV", "CONDA_PREFIX"}) {
        if (const char* root = std::getenv(variable)) {
#if defined(_WIN32)
            AddExecutableFromPath(paths, std::filesystem::path(root) / "python.exe");
            AddExecutablesFromDirectory(paths, std::filesystem::path(root));
            AddExecutableFromPath(paths, std::filesystem::path(root) / "Scripts" / "python.exe");
            AddExecutablesFromDirectory(paths, std::filesystem::path(root) / "Scripts");
#else
            AddExecutableFromPath(paths, std::filesystem::path(root) / "bin" / "python");
            AddExecutablesFromDirectory(paths, std::filesystem::path(root) / "bin");
#endif
        }
    }

    if (const char* pyenvRoot = std::getenv("PYENV_ROOT")) {
        const auto pyenvVersions = std::filesystem::path(pyenvRoot) / "versions";
        std::error_code ec;
        if (std::filesystem::is_directory(pyenvVersions, ec) && !ec) {
            for (const auto& entry : std::filesystem::directory_iterator(pyenvVersions, ec)) {
                if (ec) {
                    break;
                }
#if defined(_WIN32)
                AddExecutableFromPath(paths, entry.path() / "python.exe");
                AddExecutablesFromDirectory(paths, entry.path());
#else
                AddExecutableFromPath(paths, entry.path() / "bin" / "python");
                AddExecutablesFromDirectory(paths, entry.path() / "bin");
#endif
            }
        }
    }
}

std::string CandidateLabel(const std::string& path, const PythonRuntimeContext::Validation& validation) {
    std::string name;
    if (LooksLikePythonExecutable(path)) {
        name = ProbePythonVersion(path).value_or("Python");
    } else if (const auto version = ExtractPythonVersion(validation.libraryPath)) {
        name = "Python " + *version;
    } else {
        name = "Python Runtime";
    }

    return name + " (" + path + ")";
}

PythonRuntimeContext::Validation ValidateExplicitPythonRuntimePath(const std::string& path) {
    PythonRuntimeContext::Validation validation;
    const auto expandedPath = ExpandUserPath(path);
    validation.inputPath = expandedPath.string();

    const auto candidates = PythonLibraryCandidates(path);
    validation.attempts = candidates;
    for (const auto& candidate : candidates) {
        if (!FileIsRegular(candidate)) {
            continue;
        }

        validation.valid = true;
        validation.libraryPath = candidate;
        if (LooksLikePythonExecutable(validation.inputPath)) {
            validation.programPath = validation.inputPath;
        }
        validation.message = "Valid Python runtime: " + validation.libraryPath;
        return validation;
    }

    validation.message = "No libpython was found for " + validation.inputPath + ".";
    return validation;
}

std::vector<PythonRuntimeContext::Candidate> DiscoverRuntimeCandidatesInOrder() {
    std::vector<std::string> paths;
    AddCommonPythonExecutables(paths);

    std::vector<PythonRuntimeContext::Candidate> candidates;
    candidates.reserve(paths.size());
    for (const auto& path : paths) {
        const auto validation = ValidateExplicitPythonRuntimePath(path);
        if (!validation.valid) {
            continue;
        }

        candidates.push_back({
            .label = CandidateLabel(path, validation),
            .path = path,
            .libraryPath = validation.libraryPath,
        });
    }

    return candidates;
}

std::optional<PythonRuntimeContext::Candidate> FirstDiscoveredPythonRuntime() {
    std::vector<std::string> paths;
    AddCommonPythonExecutables(paths);

    for (const auto& path : paths) {
        const auto validation = ValidateExplicitPythonRuntimePath(path);
        if (!validation.valid) {
            continue;
        }

        return PythonRuntimeContext::Candidate{
            .label = CandidateLabel(path, validation),
            .path = path,
            .libraryPath = validation.libraryPath,
        };
    }

    return std::nullopt;
}

}  // namespace

PythonRuntimeContext::Validation PythonRuntimeContext::ValidateRuntimePath(const std::string& path) {
    if (!path.empty()) {
        return ValidateExplicitPythonRuntimePath(path);
    }

    PythonRuntimeContext::Validation validation;
    validation.inputPath = "Auto";

    const auto candidate = FirstDiscoveredPythonRuntime();
    if (!candidate.has_value()) {
        validation.message = "Auto could not find a valid Python runtime.";
        return validation;
    }

    validation = ValidateExplicitPythonRuntimePath(candidate->path);
    validation.inputPath = "Auto";
    if (validation.valid) {
        validation.message = "Auto selected " + candidate->label + ": " + validation.libraryPath;
    }
    return validation;
}

std::vector<PythonRuntimeContext::Candidate> PythonRuntimeContext::DiscoverRuntimes() {
    auto candidates = DiscoverRuntimeCandidatesInOrder();
    std::ranges::sort(candidates, {}, &PythonRuntimeContext::Candidate::label);
    return candidates;
}

}  // namespace Jetstream
