#include <iostream>
#include <thread>
#include <emscripten/proxying.h>
#include <emscripten/eventloop.h>

#include "libusb.h"
#include "interface.h"

//
// Helper macros.
//

#define DEBUG_TRACE

#define EM_PROXY(function, ...) \
    if (std::this_thread::get_id() != em_proxy_thread.get_id()) { \
        queue.proxySync(em_proxy_thread.native_handle(), [&]() { \
            function(__VA_ARGS__); \
        }); \
    } else { \
        return function(__VA_ARGS__); \
    }

#define EM_PROXY_INT(function, ...) \
    if (std::this_thread::get_id() != em_proxy_thread.get_id()) { \
        int result = 0; \
        queue.proxySync(em_proxy_thread.native_handle(), [&]() { \
            result = function(__VA_ARGS__); \
        }); \
        return result; \
    } else { \
        return function(__VA_ARGS__); \
    }

#ifdef DEBUG_TRACE
#define PRINT_DEBUG_TRACE() std::cout << "> " << __func__ << std::endl;
#else
#define PRINT_DEBUG_TRACE()
#endif

//
// Helper functions.
//

static emscripten::ProxyingQueue queue;
static std::thread em_proxy_thread;
static _Atomic bool em_proxy_started = false;

void em_proxy_init() {
    if (em_proxy_started) {
        return;
    }

    em_proxy_thread = std::thread([&]{
#ifdef DEBUG_TRACE
        std::cout << "WebUSB Thread Started (" << std::this_thread::get_id() << ")" << std::endl;
#endif
        em_proxy_started = true;
        while (em_proxy_started) {
            queue.execute();
            sched_yield();
        }
    });

    while (!em_proxy_started) {
        emscripten_sleep(10);
    }
}

void em_proxy_close() {
    if (!em_proxy_started) {
        return;
    }

    em_proxy_started = false;
    em_proxy_thread.join();
#ifdef DEBUG_TRACE
    std::cout << "WebUSB Thread Safed" << std::endl;
#endif
}

//
// Proxied methods.
//

int libusb_init(libusb_context** ctx) {
    PRINT_DEBUG_TRACE();
    em_proxy_init();

    static libusb_context _ctx;
    *ctx = &_ctx;

    EM_PROXY_INT(_libusb_init, ctx);
}

void libusb_exit(libusb_context *ctx) {
    PRINT_DEBUG_TRACE();
    em_proxy_close();
}

ssize_t libusb_get_device_list(libusb_context *ctx, libusb_device ***list) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_get_device_list, ctx, list);
}

void libusb_free_device_list(libusb_device **list, int unref_devices) {
    PRINT_DEBUG_TRACE();
    EM_PROXY(_libusb_free_device_list, list, unref_devices);
}

int libusb_get_device_descriptor(libusb_device *dev, struct libusb_device_descriptor *desc) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_get_device_descriptor, dev, desc);
}

int LIBUSB_CALL libusb_open(libusb_device *dev, libusb_device_handle **dev_handle) {
    PRINT_DEBUG_TRACE();
    libusb_device *dev_copy = new libusb_device(*dev);
    *dev_handle = (libusb_device_handle*)dev_copy;
    EM_PROXY_INT(_libusb_open, dev, nullptr);
}

void LIBUSB_CALL libusb_close(libusb_device_handle *dev_handle) {
    PRINT_DEBUG_TRACE();
    delete (libusb_device*)dev_handle;
    EM_PROXY(_libusb_close, nullptr);
}

int LIBUSB_CALL libusb_get_string_descriptor_ascii(libusb_device_handle *dev_handle,
    uint8_t desc_index, unsigned char *data, int length) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_get_string_descriptor_ascii, nullptr, desc_index, data, length);
}

int LIBUSB_CALL libusb_set_configuration(libusb_device_handle *dev_handle, int configuration) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_set_configuration, nullptr, configuration);
}

int LIBUSB_CALL libusb_claim_interface(libusb_device_handle *dev_handle, int interface_number) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_claim_interface, nullptr, interface_number);
}

int LIBUSB_CALL libusb_release_interface(libusb_device_handle *dev_handle, int interface_number) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_release_interface, nullptr, interface_number);
}

int LIBUSB_CALL libusb_control_transfer(libusb_device_handle *dev_handle,
    uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex,
    unsigned char *data, uint16_t wLength, unsigned int timeout) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_control_transfer, nullptr, request_type, bRequest, wValue, wIndex, data, wLength, timeout);
}

int LIBUSB_CALL libusb_clear_halt(libusb_device_handle *dev_handle, unsigned char endpoint) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_clear_halt, nullptr, endpoint);
}

int LIBUSB_CALL libusb_submit_transfer(struct libusb_transfer *transfer) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_submit_transfer, transfer);
}

int LIBUSB_CALL libusb_reset_device(libusb_device_handle *dev_handle) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_reset_device, nullptr);
}

int LIBUSB_CALL libusb_kernel_driver_active(libusb_device_handle *dev_handle, int interface_number) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_kernel_driver_active, nullptr, interface_number);
}

int LIBUSB_CALL libusb_cancel_transfer(struct libusb_transfer *transfer) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_cancel_transfer, transfer);
}

void LIBUSB_CALL libusb_free_transfer(struct libusb_transfer *transfer) {
    PRINT_DEBUG_TRACE();
    EM_PROXY(_libusb_free_transfer, transfer);
}

int LIBUSB_CALL libusb_handle_events_timeout(libusb_context *ctx, struct timeval *tv) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_handle_events_timeout, ctx, tv);
}

int LIBUSB_CALL libusb_handle_events_timeout_completed(libusb_context *ctx, struct timeval *tv, int *completed) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_handle_events_timeout_completed, ctx, tv, completed);
}

int LIBUSB_CALL libusb_set_interface_alt_setting(libusb_device_handle *dev_handle, int interface_number, int alternate_setting) {
    PRINT_DEBUG_TRACE();
    EM_PROXY_INT(_libusb_set_interface_alt_setting, nullptr, interface_number, alternate_setting);
}

libusb_device* LIBUSB_CALL libusb_get_device(libusb_device_handle *dev_handle) {
    return (libusb_device*)dev_handle;
}

//
// Not Proxied
//

const struct libusb_version* libusb_get_version(void) {
    PRINT_DEBUG_TRACE();
    return _libusb_get_version();
};

struct libusb_transfer * LIBUSB_CALL libusb_alloc_transfer(int iso_packets) {
    PRINT_DEBUG_TRACE();
    return _libusb_alloc_transfer(iso_packets);
}
