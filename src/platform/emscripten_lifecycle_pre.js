var cyberetherLifecycle = null;
var cyberetherGlobalListeners = [];

var addEventListener = (...args) => {
  cyberetherGlobalListeners.push(args);
  return globalThis.addEventListener(...args);
};

var removeEventListener = (...args) => {
  const [type, listener] = args;
  cyberetherGlobalListeners = cyberetherGlobalListeners.filter(([t, l]) => t !== type || l !== listener);
  return globalThis.removeEventListener(...args);
};

if (!ENVIRONMENT_IS_PTHREAD) {
  const channel = Symbol.for('cyberether.lifecycle');
  cyberetherLifecycle = Module[channel] ?? null;
  delete Module[channel];

  cyberetherLifecycle?.attach({
    release: () => {
      ABORT = true;
      const steps = [
        () => cyberetherGlobalListeners.splice(0).forEach((args) => globalThis.removeEventListener(...args)),
        () => JSEvents.removeAllEventListeners(),
        () => Object.values(GLFW3.fWindowContexts ?? {}).forEach((context) => context?.fCanvasResize?.destroy()),
        () => GLFW3.fContext && _emglfw3c_destroy(),
        () => webSockets.allocated.forEach((socket) => socket?.close()),
        () => PThread.terminateAllThreads(),
      ];
      for (const step of steps) {
        try {
          step();
        } catch (error) {
          console.error('CyberEther cleanup step failed:', error);
        }
      }
    },
    exit: (status) => {
      try {
        _emscripten_force_exit(status);
      } catch (error) {
        if (error?.name !== 'ExitStatus') {
          throw error;
        }
      }
    },
    shutdown: () => {
      _cyberether_shutdown();
    },
  });
}
