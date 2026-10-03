{
  const channel = Symbol.for('cyberether.lifecycle');
  const createRuntime = CyberEtherRuntime;
  const phases = ['starting', 'running', 'stopping'];
  const isTerminal = (state) => state === 'exited' || state === 'failed';

  const chain = (target, name, handler) => {
    let hook = target[name];
    Object.defineProperty(target, name, {
      configurable: true,
      enumerable: true,
      get: () => (...args) => {
        handler(...args);
        return hook?.apply(target, args);
      },
      set: (value) => {
        hook = value;
      },
    });
  };

  const createLifecycle = (moduleArg, factory) => {
    let runtime = null;
    let factorySettled = false;
    let cancelRequested = false;
    let exitReason = null;
    let outcome = null;
    let settle;

    const lifecycle = {
      state: 'loading',
      status: null,
      reason: null,
      cleanup: new Promise((resolve) => {
        settle = resolve;
      }),
    };

    const notify = () => {
      try {
        moduleArg.onLifecycle?.(lifecycle);
      } catch (error) {
        console.error('CyberEther lifecycle callback failed:', error);
      }
    };

    const transition = (state) => {
      if (outcome || isTerminal(lifecycle.state) || lifecycle.state === state) {
        return;
      }
      lifecycle.state = state;
      notify();
    };

    const settleFactory = (error) => {
      if (factorySettled) {
        return;
      }
      factorySettled = true;
      if (error === undefined) {
        factory.resolve(moduleArg);
      } else {
        factory.reject(error);
      }
    };

    const complete = (state, status, reason) => {
      if (outcome) {
        return;
      }
      outcome = { state, status, reason };
      try {
        runtime?.release();
      } catch (error) {
        console.error('CyberEther runtime release failed:', error);
      }
      runtime = null;
      settleFactory(reason instanceof Error ? reason : new Error(`CyberEther runtime ${state} during initialization: ${reason ?? status}`));
      lifecycle.state = state;
      lifecycle.status = status;
      lifecycle.reason = reason;
      notify();
      settle({ ...outcome });
    };

    const exitRuntime = (status, reason = null) => {
      if (!runtime) {
        complete('failed', status, new Error('CyberEther runtime is not attached.'));
        return;
      }
      exitReason = reason;
      try {
        runtime.exit(status);
      } catch (error) {
        complete('failed', status, error);
      }
    };

    lifecycle.start = (args = []) => {
      if (lifecycle.state !== 'ready') {
        throw new Error(`CyberEther cannot start while ${lifecycle.state}.`);
      }
      transition('starting');
      let result;
      try {
        result = moduleArg.callMain(args);
      } catch (error) {
        complete('failed', null, error);
        throw error;
      }
      if (result !== 0) {
        exitRuntime(result, new Error(`CyberEther failed to launch its application thread with status ${result}.`));
      }
      return result;
    };

    lifecycle.shutdown = () => {
      switch (lifecycle.state) {
        case 'loading':
          cancelRequested = true;
          transition('stopping');
          break;
        case 'ready':
          transition('stopping');
          exitRuntime(0);
          break;
        case 'starting':
        case 'running':
          try {
            runtime.shutdown();
          } catch (error) {
            complete('failed', null, error);
          }
          break;
      }
      return lifecycle.cleanup;
    };

    const controller = {
      attach: (internals) => {
        runtime = internals;
        chain(moduleArg, 'onExit', (status) => {
          complete(status === 0 ? 'exited' : 'failed', status, exitReason ?? (cancelRequested ? 'cancelled' : null));
        });
        chain(moduleArg, 'onAbort', (reason) => {
          complete('failed', null, reason);
        });
      },
      report: (phase) => {
        if (phases[phase]) {
          transition(phases[phase]);
        }
      },
      resolved: () => {
        if (factorySettled) {
          return;
        }
        if (cancelRequested) {
          settleFactory();
          exitRuntime(0);
        } else {
          settleFactory();
          transition('ready');
        }
      },
      rejected: (error) => {
        if (factorySettled) {
          return;
        }
        complete('failed', null, error);
      },
    };

    return { lifecycle, controller };
  };

  CyberEtherRuntime = (moduleArg = {}) => {
    const factory = {};
    const pending = new Promise((resolve, reject) => {
      factory.resolve = resolve;
      factory.reject = reject;
    });

    const { lifecycle, controller } = createLifecycle(moduleArg, factory);
    moduleArg.cyberether = lifecycle;
    moduleArg[channel] = controller;

    let runtime;
    try {
      runtime = createRuntime(moduleArg);
    } catch (error) {
      runtime = Promise.reject(error);
    }
    runtime.then(controller.resolved, controller.rejected);
    return pending;
  };
}
