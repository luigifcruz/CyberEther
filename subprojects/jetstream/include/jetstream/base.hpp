#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include "type.hpp"

namespace Jetstream {

class Module;

using Graph = std::vector<std::shared_ptr<Module>>;

typedef struct {
    Launch launch;
    Graph deps;
} Policy;

class Scheduler {
public:
    explicit Scheduler(Graph& d) : deps(d) {};
    virtual ~Scheduler() = default;

protected:
    virtual Result start() = 0;
    virtual Result end() = 0;

    virtual Result compute() = 0;
    virtual Result barrier() = 0;

    virtual Result underlyingCompute() = 0;

    Graph& deps;

    std::atomic<Result> result{SUCCESS};
};

class Async : public Scheduler {
public:
    explicit Async(Graph& d) : Scheduler(d) {};
    ~Async() = default;

protected:
    Result start();
    Result end();

    Result compute();
    Result barrier();

private:
    std::mutex m;
    std::thread worker;
    bool mailbox{false};
    bool discard{false};
    std::condition_variable access;

    friend class Module;
};

class Sync : public Scheduler {
public:
    explicit Sync(Graph& d) : Scheduler(d) {};
    ~Sync() = default;

protected:
    Result start();
    Result end();

    Result compute();
    Result barrier();

private:
    std::mutex m;
    std::atomic<bool> mailbox{false};

    friend class Module;
};

class Module : public Async, public Sync {
public:
    explicit Module(Policy&);
    virtual ~Module();

    Result compute();
    Result barrier();
    Result present();

protected:
    Launch launch;

    virtual Result underlyingPresent() = 0;
};

class Engine : public Graph {
public:
    Result compute();
    Result present();

private:
    std::mutex m;
    std::condition_variable access;
    std::atomic<bool> waiting{false};
};

} // namespace Jetstream

#endif
