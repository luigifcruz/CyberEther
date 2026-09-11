#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include "jetstream/platform.hh"

using namespace Jetstream;

namespace {

struct CapturedResource {
    std::promise<std::thread::id>& destroyed;
    std::shared_future<void> release;
    std::thread::id& workerId;

    CapturedResource(std::promise<std::thread::id>& destroyed,
                     const std::shared_future<void>& release,
                     std::thread::id& workerId)
        : destroyed(destroyed), release(release), workerId(workerId) {}

    ~CapturedResource() {
        destroyed.set_value(std::this_thread::get_id());
    }
};

}  // namespace

TEST_CASE("Worker releases owning captures on its own thread before joining",
          "[core][platform][worker][lifetime]") {
    std::promise<std::thread::id> destroyed;
    auto destroyedFuture = destroyed.get_future();
    std::promise<void> release;
    std::thread::id workerId;
    Platform::WorkerThread worker;

    // Keep the callable small enough for std::function's inline storage.
    worker.start([resource = std::make_shared<CapturedResource>(
                      destroyed, release.get_future().share(), workerId)] {
        resource->release.wait();
        resource->workerId = std::this_thread::get_id();
    });
    release.set_value();

    const auto status = destroyedFuture.wait_for(std::chrono::seconds(2));
    worker.join();

    CHECK(status == std::future_status::ready);
    CHECK(destroyedFuture.get() == workerId);
}
