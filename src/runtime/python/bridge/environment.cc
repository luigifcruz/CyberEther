#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/convert.hh"

#include "jetstream/flowgraph_environment.hh"
#include "jetstream/logger.hh"

namespace Jetstream {

using namespace CPython;

PyObject* Bridge::createEnvironmentDict() {
    if (!globals) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python environment without globals.");
        return nullptr;
    }

    auto* factory = PyDict_GetItemString(globals, "_jetstream_create_environment");
    if (!factory || !PyCallable_Check(factory)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime helper '_jetstream_create_environment' is unavailable.");
        return nullptr;
    }

    auto* dict = PyObject_CallFunctionObjArgs(factory);
    if (!dict) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python environment dict.");
        return nullptr;
    }

    Py_IncRef(dict);
    environmentDict = dict;
    environmentSynced = false;

    return dict;
}

void Bridge::refreshEnvironment() {
    if (!environmentDict || !environment) {
        return;
    }

    const U64 epoch = environment->epoch();
    if (environmentSynced && epoch == environmentEpoch) {
        return;
    }

    std::vector<std::pair<std::string, U64>> versions;
    if (environment->versions(versions) != Result::SUCCESS) {
        return;
    }

    if (!environmentSynced) {
        PyDict_Clear(environmentDict);
        environmentVersions.clear();
    }

    auto* converters = valueConverterTable();

    std::vector<std::string> changed;
    for (const auto& [key, version] : versions) {
        const auto it = environmentVersions.find(key);
        if (it != environmentVersions.end() && it->second == version) {
            continue;
        }

        Parser::Map data;
        if (environment->get(key, data) != Result::SUCCESS) {
            continue;
        }

        auto* object = AnyToPyObject(std::any(std::move(data)), converters);
        if (!object) {
            JST_TRACE("[RUNTIME_CONTEXT_PYTHON] Skipping unsupported environment value '{}'.", key);
            (void)ClearPythonError();
            if (PyDict_DelItemString(environmentDict, key.c_str()) != 0) {
                (void)ClearPythonError();
            }
            continue;
        }

        if (PyDict_SetItemString(environmentDict, key.c_str(), object) != 0) {
            Py_DecRef(object);
            (void)ClearPythonError();
            continue;
        }
        Py_DecRef(object);
        changed.push_back(key);
    }

    std::unordered_map<std::string, U64> current;
    current.reserve(versions.size());
    for (auto& [key, version] : versions) {
        current.emplace(std::move(key), version);
    }

    for (const auto& [key, version] : environmentVersions) {
        if (current.contains(key)) {
            continue;
        }

        if (PyDict_DelItemString(environmentDict, key.c_str()) != 0) {
            (void)ClearPythonError();
        }
    }

    environmentVersions = std::move(current);

    trackEnvironment(changed);

    environmentEpoch = epoch;
    environmentSynced = true;
}

void Bridge::trackEnvironment(const std::vector<std::string>& keys) {
    if (!environmentDict || !globals || keys.empty()) {
        return;
    }

    auto* track = PyDict_GetItemString(globals, "_jetstream_track_environment");
    if (!track || !PyCallable_Check(track)) {
        return;
    }

    auto* keysTuple = PyTuple_New(static_cast<Py_ssize_t>(keys.size()));
    if (!keysTuple) {
        (void)ClearPythonError();
        return;
    }

    for (std::size_t index = 0; index < keys.size(); ++index) {
        auto* key = PyUnicode_FromString(keys[index].c_str());
        if (!key) {
            Py_DecRef(keysTuple);
            (void)ClearPythonError();
            return;
        }

        if (PyTuple_SetItem(keysTuple, static_cast<Py_ssize_t>(index), key) != 0) {
            Py_DecRef(keysTuple);
            (void)ClearPythonError();
            return;
        }
    }

    auto* result = PyObject_CallFunctionObjArgs(track, environmentDict, keysTuple);
    Py_DecRef(keysTuple);
    if (!result) {
        (void)ClearPythonError();
        return;
    }
    Py_DecRef(result);
}

void Bridge::flushEnvironment() {
    if (!environmentDict || !environment || !globals) {
        return;
    }

    auto* classify = PyDict_GetItemString(globals, "_jetstream_classify_attribute");
    if (!classify || !PyCallable_Check(classify)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime helper '_jetstream_classify_attribute' is unavailable.");
        return;
    }

    auto* consume = PyDict_GetItemString(globals, "_jetstream_consume_dirty_environment");
    if (!consume || !PyCallable_Check(consume)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime helper '_jetstream_consume_dirty_environment' is unavailable.");
        return;
    }

    auto* dirtyKeys = PyObject_CallFunctionObjArgs(consume, environmentDict);
    if (!dirtyKeys) {
        (void)ClearPythonError();
        return;
    }

    auto restoreCanonical = [&](const char* keyStr, const bool hasExisting, const Parser::Map& existingData) {
        if (!hasExisting) {
            if (PyDict_DelItemString(environmentDict, keyStr) != 0) {
                (void)ClearPythonError();
            }
            return;
        }

        auto* object = AnyToPyObject(std::any(existingData), valueConverterTable());
        if (!object) {
            (void)ClearPythonError();
            return;
        }

        if (PyDict_SetItemString(environmentDict, keyStr, object) != 0) {
            (void)ClearPythonError();
        }
        Py_DecRef(object);
    };

    const auto count = PySequence_Size(dirtyKeys);
    std::vector<std::string> processed;
    processed.reserve(static_cast<std::size_t>(count > 0 ? count : 0));
    for (Py_ssize_t index = 0; index < count; ++index) {
        auto* key = PySequence_GetItem(dirtyKeys, index);
        if (!key) {
            (void)ClearPythonError();
            continue;
        }

        const char* keyStr = PyUnicode_AsUTF8(key);
        if (!keyStr) {
            (void)ClearPythonError();
            Py_DecRef(key);
            continue;
        }
        processed.emplace_back(keyStr);

        Parser::Map existingData;
        bool hasExisting = environment->has(keyStr);
        if (hasExisting && environment->get(keyStr, existingData) != Result::SUCCESS) {
            hasExisting = false;
        }

        auto* value = PyDict_GetItemString(environmentDict, keyStr);
        if (!value) {
            restoreCanonical(keyStr, hasExisting, existingData);
            Py_DecRef(key);
            continue;
        }

        std::any existing;
        if (hasExisting) {
            existing = existingData;
        }

        std::any converted;
        if (PyObjectToAny(classify, value, existing, converted) != Result::SUCCESS) {
            JST_WARN("[RUNTIME_CONTEXT_PYTHON] Ignoring unsupported environment value '{}'.", keyStr);
            restoreCanonical(keyStr, hasExisting, existingData);
            Py_DecRef(key);
            continue;
        }

        if (converted.type() != typeid(Parser::Map)) {
            JST_WARN("[RUNTIME_CONTEXT_PYTHON] Environment value '{}' must be a mapping.", keyStr);
            restoreCanonical(keyStr, hasExisting, existingData);
            Py_DecRef(key);
            continue;
        }

        const auto& convertedData = std::any_cast<const Parser::Map&>(converted);
        if (!hasExisting || !AnyDeepEquals(converted, existing)) {
            if (environment->set(keyStr, convertedData) != Result::SUCCESS) {
                JST_WARN("[RUNTIME_CONTEXT_PYTHON] Can't publish environment value '{}'.", keyStr);
                restoreCanonical(keyStr, hasExisting, existingData);
            }
        }

        Py_DecRef(key);
    }

    trackEnvironment(processed);

    Py_DecRef(dirtyKeys);
}

void Bridge::destroyEnvironmentDict() {
    if (environmentDict) {
        Py_DecRef(environmentDict);
        environmentDict = nullptr;
    }
    environmentSynced = false;
    environmentVersions.clear();
}

}  // namespace Jetstream
