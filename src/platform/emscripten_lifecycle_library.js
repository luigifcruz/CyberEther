addToLibrary({
  cyberether_lifecycle_report__deps: [
    'emscripten_force_exit',
    'emglfw3c_destroy',
    '$GLFW3',
    '$JSEvents',
    '$webSockets',
    '$PThread',
  ],
  cyberether_lifecycle_report__proxy: 'async',
  cyberether_lifecycle_report: (phase) => {
    cyberetherLifecycle?.report(phase);
  },
});
