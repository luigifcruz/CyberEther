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

    std::vector<std::string> keys;
    if (environment->keys(keys) != Result::SUCCESS) {
        return;
    }

    PyDict_Clear(environmentDict);

    auto* converters = valueConverterTable();

    for (const auto& key : keys) {
        Parser::Map data;
        if (environment->get(key, data) != Result::SUCCESS) {
            continue;
        }

        auto* object = AnyToPyObject(std::any(std::move(data)), converters);
        if (!object) {
            JST_TRACE("[RUNTIME_CONTEXT_PYTHON] Skipping unsupported environment value '{}'.", key);
            (void)ClearPythonError();
            continue;
        }

        if (PyDict_SetItemString(environmentDict, key.c_str(), object) != 0) {
            Py_DecRef(object);
            (void)ClearPythonError();
            continue;
        }
        Py_DecRef(object);
    }

    trackEnvironment();

    environmentEpoch = epoch;
    environmentSynced = true;
}

void Bridge::trackEnvironment() {
    if (!environmentDict || !globals) {
        return;
    }

    auto* track = PyDict_GetItemString(globals, "_jetstream_track_environment");
    if (!track || !PyCallable_Check(track)) {
        return;
    }

    auto* result = PyObject_CallFunctionObjArgs(track, environmentDict);
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

    if (count > 0) {
        trackEnvironment();
    }

    Py_DecRef(dirtyKeys);
}

void Bridge::destroyEnvironmentDict() {
    if (environmentDict) {
        Py_DecRef(environmentDict);
        environmentDict = nullptr;
    }
    environmentSynced = false;
}

}  // namespace Jetstream
