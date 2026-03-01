#include "jetstream/compositor.hh"
#include "jetstream/detail/compositor_impl.hh"

namespace Jetstream {

std::shared_ptr<Compositor::Impl> DefaultCompositorFactory();

Compositor::Compositor(const CompositorType& type) {
    switch (type) {
        case CompositorType::DEFAULT:
            impl = DefaultCompositorFactory();
            break;
        default:
            JST_FATAL("[COMPOSITOR] Unknown compositor type.");
            throw Result::FATAL;
    }
}

Result Compositor::create(const std::shared_ptr<Instance>& instance,
                          const std::shared_ptr<Render::Window>& render,
                          const std::shared_ptr<Viewport::Generic>& viewport) {
    impl->instance = instance;
    impl->render = render;
    impl->viewport = viewport;

    impl->startWorker();
    return impl->create();
}

Result Compositor::destroy() {
    impl->stopWorker();
    return impl->destroy();
}

Result Compositor::present() {
    return impl->present();
}

Result Compositor::poll() {
    return impl->poll();
}

}  // namespace Jetstream
