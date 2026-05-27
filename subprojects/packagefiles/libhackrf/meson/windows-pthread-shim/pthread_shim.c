#ifdef _WIN32
#include "pthread.h"

struct pthread_shim_start_ctx {
    void *(*start_routine)(void *);
    void *arg;
};

static unsigned __stdcall pthread_shim_start(void *arg) {
    struct pthread_shim_start_ctx *ctx = (struct pthread_shim_start_ctx *)arg;
    void *ret = ctx->start_routine(ctx->arg);
    HeapFree(GetProcessHeap(), 0, ctx);
    return (unsigned)(uintptr_t)ret;
}

int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr) {
    (void)attr;
    InitializeCriticalSection(mutex);
    return 0;
}

int pthread_mutex_destroy(pthread_mutex_t *mutex) {
    DeleteCriticalSection(mutex);
    return 0;
}

int pthread_mutex_lock(pthread_mutex_t *mutex) {
    EnterCriticalSection(mutex);
    return 0;
}

int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    LeaveCriticalSection(mutex);
    return 0;
}

int pthread_cond_init(pthread_cond_t *cond, const void *attr) {
    (void)attr;
    InitializeConditionVariable(cond);
    return 0;
}

int pthread_cond_destroy(pthread_cond_t *cond) {
    (void)cond;
    return 0;
}

int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    BOOL ok = SleepConditionVariableCS(cond, mutex, INFINITE);
    return ok ? 0 : 1;
}

int pthread_cond_broadcast(pthread_cond_t *cond) {
    WakeAllConditionVariable(cond);
    return 0;
}

int pthread_create(pthread_t *thread, const void *attr, void *(*start_routine)(void *), void *arg) {
    (void)attr;
    struct pthread_shim_start_ctx *ctx = (struct pthread_shim_start_ctx *)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*ctx));
    if (ctx == NULL) {
        return 1;
    }
    ctx->start_routine = start_routine;
    ctx->arg = arg;

    uintptr_t th = _beginthreadex(NULL, 0, pthread_shim_start, ctx, 0, NULL);
    if (th == 0) {
        HeapFree(GetProcessHeap(), 0, ctx);
        return 1;
    }
    *thread = th;
    return 0;
}

int pthread_join(pthread_t thread, void **retval) {
    DWORD wait_res = WaitForSingleObject((HANDLE)thread, INFINITE);
    if (wait_res != WAIT_OBJECT_0) {
        return 1;
    }

    if (retval != NULL) {
        DWORD code = 0;
        if (!GetExitCodeThread((HANDLE)thread, &code)) {
            CloseHandle((HANDLE)thread);
            return 1;
        }
        *retval = (void *)(uintptr_t)code;
    }

    CloseHandle((HANDLE)thread);
    return 0;
}

#endif
