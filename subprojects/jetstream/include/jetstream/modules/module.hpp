#ifndef JETSTREAM_MODULE_H
#define JETSTREAM_MODULE_H

#include "jetstream/type.hpp"

namespace Jetstream {

class Module {
public:
    explicit Module(IO & input) : io_input(input) {};
    virtual ~Module() = default;
    virtual Result compute() = 0;
    virtual Result present() = 0;

    Variant output(const std::string & name) {
        return io_output[name];
    }

protected:
    IO& io_input;
    IO io_output;

    template<typename T>
    T getInput(const std::string & name) {
        return std::get<T>(io_input[name]);
    }

    void setOutput(const std::string & name, const Variant & var) {
        io_output[name] = var;
    }
};

} // namespace Jetstream

#endif
