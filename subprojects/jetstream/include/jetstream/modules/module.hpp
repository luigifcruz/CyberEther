#ifndef JETSTREAM_MODULE_H
#define JETSTREAM_MODULE_H

#include "jetstream/type.hpp"

namespace Jetstream {

class Module {
public:
    explicit Module(Connections & input) : connections(input) {};
    virtual ~Module() = default;

    virtual Result compute() = 0;
    virtual Result present() = 0;

    DataContainer output(const std::string & name) {
        return connections[name];
    }

protected:
    Connections& connections;

    template<typename T>
    T getInput(const std::string & name) {
        return std::get<T>(connections[name]);
    }

    void setOutput(const std::string & name, const DataContainer & var) {
        connections[name] = var;
    }
};

} // namespace Jetstream

#endif
