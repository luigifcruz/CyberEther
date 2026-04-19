#include "jetstream/detail/compositor_impl.hh"

namespace Jetstream {

void Compositor::Impl::startWorker() {
    workerRunning = true;
    workerThread = std::thread([this]() {
        while (true) {
            Command cmd;
            {
                std::unique_lock<std::mutex> lock(commandPendingQueueMutex);
                commandQueueNotEmpty.wait(lock, [this] {
                    return !commandPendingQueue.empty() || !workerRunning;
                });

                if (!workerRunning && commandPendingQueue.empty()) {
                    break;
                }

                cmd = std::move(commandPendingQueue.front());
                commandPendingQueue.pop();
            }

            cmd.result = cmd.fn();

            if (cmd.result == Result::ERROR ||
                cmd.result == Result::INCOMPLETE) {
                cmd.message = JST_LOG_LAST_ERROR();
            } else if (cmd.result == Result::WARNING) {
                cmd.message = JST_LOG_LAST_WARNING();
            } else if (cmd.result == Result::FATAL) {
                cmd.message = JST_LOG_LAST_FATAL();
            }

            cmd.fn = nullptr;

            {
                std::lock_guard<std::mutex> lock(commandCompletedQueueMutex);
                commandCompletedQueue.push(std::move(cmd));
            }
        }

        return Result::SUCCESS;
    });
}

void Compositor::Impl::stopWorker() {
    {
        std::lock_guard<std::mutex> lock(commandPendingQueueMutex);
        workerRunning = false;
    }
    commandQueueNotEmpty.notify_all();
    if (workerThread.joinable()) {
        workerThread.join();
    }
}

void Compositor::Impl::enqueue(std::function<Result()> fn, bool silent) {
    std::lock_guard<std::mutex> lock(commandPendingQueueMutex);
    if (!workerRunning) {
        return;
    }

    commandPendingQueue.push({std::move(fn), silent});
    commandQueueNotEmpty.notify_one();
}

bool Compositor::Impl::dequeue(Command& command) {
    std::lock_guard<std::mutex> lock(commandCompletedQueueMutex);

    if (commandCompletedQueue.empty()) {
        return false;
    }

    command = std::move(commandCompletedQueue.front());
    commandCompletedQueue.pop();
    return true;
}

}  // namespace Jetstream
