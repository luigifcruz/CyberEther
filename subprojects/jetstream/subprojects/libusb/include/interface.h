#ifndef INTERFACE_H
#define INTERFACE_H

#include "libusb.h"

struct libusb_context {};

struct libusb_device {
    int id;  
};

struct libusb_device_handle {
    int id;  
};

const struct libusb_version* _libusb_get_version(void);

int _libusb_init(libusb_context**);

ssize_t _libusb_get_device_list(libusb_context *, libusb_device ***);

int _libusb_get_device_descriptor(libusb_device *, struct libusb_device_descriptor *);

void _libusb_free_device_list(libusb_device **, int);

int LIBUSB_CALL _libusb_open(libusb_device *, libusb_device_handle **);

void LIBUSB_CALL _libusb_close(libusb_device_handle *);

int LIBUSB_CALL _libusb_get_string_descriptor_ascii(libusb_device_handle *, uint8_t, unsigned char *, int);

int LIBUSB_CALL _libusb_set_configuration(libusb_device_handle *, int);

int LIBUSB_CALL _libusb_claim_interface(libusb_device_handle *, int);

int LIBUSB_CALL _libusb_release_interface(libusb_device_handle *, int);

int LIBUSB_CALL _libusb_control_transfer(libusb_device_handle *, uint8_t, uint8_t, uint16_t, uint16_t, unsigned char *, uint16_t, unsigned int);

struct libusb_transfer * LIBUSB_CALL _libusb_alloc_transfer(int);

int LIBUSB_CALL _libusb_clear_halt(libusb_device_handle *, unsigned char);

int LIBUSB_CALL _libusb_submit_transfer(struct libusb_transfer *);

int LIBUSB_CALL _libusb_reset_device(libusb_device_handle *);

int LIBUSB_CALL _libusb_kernel_driver_active(libusb_device_handle *, int);

int LIBUSB_CALL _libusb_cancel_transfer(struct libusb_transfer *);

void LIBUSB_CALL _libusb_free_transfer(struct libusb_transfer *);

int LIBUSB_CALL _libusb_set_interface_alt_setting(libusb_device_handle*, int, int);

int LIBUSB_CALL _libusb_handle_events_timeout(libusb_context *, struct timeval *);

int LIBUSB_CALL _libusb_handle_events_timeout_completed(libusb_context *, struct timeval *, int *);

#endif
