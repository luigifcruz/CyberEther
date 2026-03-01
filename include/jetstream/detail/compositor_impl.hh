#ifndef JETSTREAM_COMPOSITOR_IMPL_HH
#define JETSTREAM_COMPOSITOR_IMPL_HH

#include <queue>
#include <string>
#include <mutex>
#include <thread>
#include <deque>
#include <functional>
#include <condition_variable>

#include "jetstream/compositor.hh"
#include "jetstream/instance.hh"

namespace Jetstream {

struct Compositor::Impl {
 public:
    virtual ~Impl() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;

    virtual Result present() = 0;
    virtual Result poll() = 0;

 protected:
    std::shared_ptr<Instance> instance;
    std::shared_ptr<Render::Window> render;
    std::shared_ptr<Viewport::Generic> viewport;

    // Worker thread.
    std::thread workerThread;
    bool workerRunning = false;

    // Command queue.
    struct Command {
        std::function<Result()> fn;
        bool silent = false;
        Result result = Result::SUCCESS;
        std::string message;
    };
    std::queue<Command> commandPendingQueue;
    std::queue<Command> commandCompletedQueue;
    std::mutex commandPendingQueueMutex;
    std::mutex commandCompletedQueueMutex;
    std::condition_variable commandQueueNotEmpty;

    void startWorker();
    void stopWorker();
    void enqueue(std::function<Result()> fn, bool silent = false);
    bool dequeue(Command& command);

    friend class Compositor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_HH
