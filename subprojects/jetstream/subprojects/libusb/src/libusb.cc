#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <mutex>

#include <emscripten.h>
#include <emscripten/val.h>
#include <emscripten/threading.h>

#include "interface.h"

// TODO: Lots of opportunities for refactor in this file.

using namespace emscripten;

#define LIBUSB_MANUFACTURER_ID  ((uint8_t)1)
#define LIBUSB_PRODUCT_ID       ((uint8_t)2)
#define LIBUSB_SN_ID            ((uint8_t)3)

std::mutex mutex;
std::vector<struct libusb_transfer*> transfers;
std::vector<struct libusb_transfer*> staging;

val create_out_buffer(uint8_t* buffer, size_t size) {
    val buf = val::global("Uint8Array").new_(size);
    val tmp = val(typed_memory_view(size, buffer));
    buf.call<val>("set", val(tmp));
    return buf;
}

int pick_device() {
    val usb = val::global("navigator")["usb"];

    val filter = val::object();
    filter.set("filters", val::array());
    val dev = usb.call<val>("requestDevice", filter).await();

    if (!dev.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    return LIBUSB_SUCCESS;
}

const struct libusb_version* _libusb_get_version(void) {
    static struct libusb_version info = {1, 0, 24, 0};
    return &info;
};

int _libusb_init(libusb_context**) {
    val navigator = val::global("navigator");

    if (!navigator["usb"].as<bool>()) {
        std::cerr << "WebUSB not supported by browser." << std::endl;
        return LIBUSB_ERROR_NOT_SUPPORTED;
    }

    return LIBUSB_SUCCESS;
};

ssize_t _libusb_get_device_list(libusb_context *ctx, libusb_device ***list) {
    val devices = val::global("navigator")["usb"].call<val>("getDevices").await();
    int available = devices["length"].as<int>();

    if (available == 0) {
        return LIBUSB_ERROR_NO_DEVICE;
    }

    *list = new libusb_device*[available + 1];
    for (int i = 0; i < available; i++) {
        libusb_device* device = new libusb_device[1];
        device->id = i;
        (*list)[i] = device;
    }
    (*list)[available] = nullptr;

    val::global().set("devices", devices);

    return available;
}

void _libusb_free_device_list(libusb_device **list, int unref_devices) {
    for (int i = 0; i < unref_devices + 1; i++) {
        delete list[i];
    }
    delete[] list;
}

int _libusb_get_device_descriptor(libusb_device *dev, struct libusb_device_descriptor *desc) {
    val devices = val::global("devices");

    if (!devices.as<bool>())
        return LIBUSB_ERROR_INVALID_PARAM;

    val d = devices[dev->id];

    if (!d.as<bool>())
        return LIBUSB_ERROR_INVALID_PARAM;

    desc->bLength = LIBUSB_DT_DEVICE_SIZE;
    desc->bDescriptorType = LIBUSB_DT_DEVICE;
    desc->bcdUSB = d["usbVersionMajor"].as<uint8_t>() << 8 |
                   d["usbVersionMinor"].as<uint8_t>() << 0 ;
    desc->bDeviceClass = d["deviceClass"].as<uint8_t>();
    desc->bDeviceSubClass = d["deviceSubclass"].as<uint8_t>();
    desc->bDeviceProtocol = d["deviceProtocol"].as<uint8_t>();
    desc->bMaxPacketSize0 = 64;
    desc->idVendor = d["vendorId"].as<uint16_t>();
    desc->idProduct = d["productId"].as<uint16_t>();
    desc->bcdDevice = d["deviceVersionMajor"].as<uint8_t>()    << 8 |
                      d["deviceVersionMinor"].as<uint8_t>()    << 4 |
                      d["deviceVersionSubminor"].as<uint8_t>() << 0 ;
    desc->iManufacturer = LIBUSB_MANUFACTURER_ID;
    desc->iProduct = LIBUSB_PRODUCT_ID;
    desc->iSerialNumber = LIBUSB_SN_ID;
    desc->bNumConfigurations = d["configurations"]["length"].as<uint8_t>();

    return LIBUSB_SUCCESS;
}

int LIBUSB_CALL _libusb_open(libusb_device *dev, libusb_device_handle **dev_handle) {
    val device = val::global("devices")[dev->id];

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    device.call<val>("open").await();

    val::global().set("device", device);

    return LIBUSB_SUCCESS;
}

void LIBUSB_CALL _libusb_close(libusb_device_handle *dev_handle) {
    val device = val::global("device");

    if (!device.as<bool>())
        return;

    device.call<val>("close").await();
}

int LIBUSB_CALL _libusb_get_string_descriptor_ascii(libusb_device_handle *dev_handle,
    uint8_t desc_index, unsigned char *data, int length) {
    val device = val::global("device");

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    std::string str;
    switch (desc_index) {
        case LIBUSB_PRODUCT_ID:
            str = device["productName"].as<std::string>();
            std::copy(str.begin(), str.end(), data);
            return str.size();
        case LIBUSB_MANUFACTURER_ID:
            str = device["manufacturerName"].as<std::string>();
            std::copy(str.begin(), str.end(), data);
            return str.size();
        case LIBUSB_SN_ID:
            str = device["serialNumber"].as<std::string>();
            std::copy(str.begin(), str.end(), data);
            return str.size();
    }

    return LIBUSB_ERROR_INVALID_PARAM;
}

int LIBUSB_CALL _libusb_set_configuration(libusb_device_handle *dev_handle, int configuration) {
    val device = val::global("device");

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    device.call<val>("selectConfiguration", configuration).await();

    return LIBUSB_SUCCESS;
}

int LIBUSB_CALL _libusb_claim_interface(libusb_device_handle *dev_handle, int interface_number) {
    val device = val::global("device");

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    device.call<val>("claimInterface", interface_number).await();

    return LIBUSB_SUCCESS;
}

int LIBUSB_CALL _libusb_release_interface(libusb_device_handle *dev_handle, int interface_number) {
    val device = val::global("device");

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    device.call<val>("releaseInterface", interface_number).await();

    return LIBUSB_SUCCESS;
}

int LIBUSB_CALL _libusb_control_transfer(libusb_device_handle *dev_handle,
    uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex,
    unsigned char *data, uint16_t wLength, unsigned int timeout) {
    val device = val::global("device");

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    val setup = val::object();
    setup.set("request", bRequest);
    setup.set("value", wValue);
    setup.set("index", wIndex);

    switch ((request_type & 0x31)) {
        case LIBUSB_RECIPIENT_DEVICE:
            setup.set("recipient", std::string("device"));
            break;
        case LIBUSB_RECIPIENT_INTERFACE:
            setup.set("recipient", std::string("interface"));
            break;
        case LIBUSB_RECIPIENT_ENDPOINT:
            setup.set("recipient", std::string("endpoint"));
            break;
        case LIBUSB_RECIPIENT_OTHER:
            setup.set("recipient", std::string("other"));
            break;
    }

    switch ((request_type & 0x60)) {
        case LIBUSB_REQUEST_TYPE_STANDARD:
            setup.set("requestType", std::string("standard"));
            break;
        case LIBUSB_REQUEST_TYPE_CLASS:
            setup.set("requestType", std::string("class"));
            break;
        case LIBUSB_REQUEST_TYPE_VENDOR:
            setup.set("requestType", std::string("vendor"));
            break;
        case LIBUSB_REQUEST_TYPE_RESERVED:
            return LIBUSB_ERROR_INVALID_PARAM;
    }

    if ((request_type & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_IN) {
        val res = device.call<val>("controlTransferIn", setup, wLength).await();

        if (res["status"].as<std::string>().compare("ok"))
            return LIBUSB_ERROR_IO;

        auto buf = res["data"]["buffer"].as<std::string>();
        std::copy(buf.begin(), buf.end(), data);

        return res["data"]["buffer"]["byteLength"].as<int>();
    }

    if ((request_type & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_OUT) {
        auto buf = create_out_buffer(data, wLength);
        val res = device.call<val>("controlTransferOut", setup, buf).await();

        if (res["status"].as<std::string>().compare("ok"))
            return LIBUSB_ERROR_IO;

        return res["bytesWritten"].as<int>();
    }

    return LIBUSB_ERROR_OTHER;
}

struct libusb_transfer * LIBUSB_CALL _libusb_alloc_transfer(int iso_packets) {
    size_t alloc_size =
		sizeof(struct libusb_transfer) +
		(sizeof(struct libusb_iso_packet_descriptor) * (size_t)iso_packets);

    return (struct libusb_transfer*)malloc(alloc_size);
}

int LIBUSB_CALL _libusb_clear_halt(libusb_device_handle *dev_handle, unsigned char endpoint) {
    val device = val::global("device");

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    std::string direction = ((endpoint & LIBUSB_ENDPOINT_DIR_MASK) & LIBUSB_ENDPOINT_OUT) ? "out" : "in";
    unsigned char num = endpoint & ~LIBUSB_ENDPOINT_DIR_MASK;

    device.call<val>("clearHalt", direction, num).await();

    return LIBUSB_SUCCESS;
}

int LIBUSB_CALL _libusb_submit_transfer(struct libusb_transfer *transfer) {
    val device = val::global("device");

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    {
        std::lock_guard<std::mutex> lock(mutex);
        staging.push_back(transfer);
    }

    unsigned char num = transfer->endpoint & ~LIBUSB_ENDPOINT_DIR_MASK;
    if ((transfer->endpoint & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_IN) {
        switch(transfer->type) {
            case LIBUSB_TRANSFER_TYPE_BULK: {
                val res = device.call<val>("transferIn", num, transfer->length).await();

                if (res["status"].as<std::string>().compare("ok"))
                    return LIBUSB_ERROR_IO;

                auto buf = res["data"]["buffer"].as<std::string>();
                std::copy(buf.begin(), buf.end(), transfer->buffer);

                transfer->actual_length = res["data"]["buffer"]["byteLength"].as<int>();
                if (transfer->status != LIBUSB_TRANSFER_CANCELLED) {
                    transfer->status = LIBUSB_TRANSFER_COMPLETED;
                }

                return LIBUSB_SUCCESS;
            }
            case LIBUSB_TRANSFER_TYPE_BULK_STREAM:
                std::cout << "Not implemented: IN LIBUSB_TRANSFER_TYPE_BULK_STREAM" << std::endl;
                return LIBUSB_ERROR_NOT_SUPPORTED;
            case LIBUSB_TRANSFER_TYPE_CONTROL:
                std::cout << "Not implemented: IN LIBUSB_TRANSFER_TYPE_CONTROL" << std::endl;
                return LIBUSB_ERROR_NOT_SUPPORTED;
            case LIBUSB_TRANSFER_TYPE_INTERRUPT:
                std::cout << "Not implemented: IN LIBUSB_TRANSFER_TYPE_INTERRUPT" << std::endl;
                return LIBUSB_ERROR_NOT_SUPPORTED;
            case LIBUSB_TRANSFER_TYPE_ISOCHRONOUS:
                std::cout << "Not implemented: IN LIBUSB_TRANSFER_TYPE_ISOCHRONOUS" << std::endl;
                return LIBUSB_ERROR_NOT_SUPPORTED;
        }
    }

    if ((transfer->endpoint & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_OUT) {
        switch(transfer->type) {
            case LIBUSB_TRANSFER_TYPE_BULK:
                std::cout << "Not implemented: OUT LIBUSB_TRANSFER_TYPE_BULK" << std::endl;
                return LIBUSB_ERROR_NOT_SUPPORTED;
            case LIBUSB_TRANSFER_TYPE_BULK_STREAM:
                std::cout << "Not implemented: OUT LIBUSB_TRANSFER_TYPE_BULK_STREAM" << std::endl;
                return LIBUSB_ERROR_NOT_SUPPORTED;
            case LIBUSB_TRANSFER_TYPE_CONTROL:
                std::cout << "Not implemented: OUT LIBUSB_TRANSFER_TYPE_CONTROL" << std::endl;
                return LIBUSB_ERROR_NOT_SUPPORTED;
            case LIBUSB_TRANSFER_TYPE_INTERRUPT:
                std::cout << "Not implemented: OUT LIBUSB_TRANSFER_TYPE_INTERRUPT" << std::endl;
                return LIBUSB_ERROR_NOT_SUPPORTED;
            case LIBUSB_TRANSFER_TYPE_ISOCHRONOUS:
                std::cout << "Not implemented: OUT LIBUSB_TRANSFER_TYPE_ISOCHRONOUS" << std::endl;
                return LIBUSB_ERROR_NOT_SUPPORTED;
        }
    }

    return LIBUSB_ERROR_OTHER;
}

int LIBUSB_CALL _libusb_reset_device(libusb_device_handle *dev_handle) {
    val device = val::global("device");

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    device.call<val>("reset").await();

    return LIBUSB_SUCCESS;
}

int LIBUSB_CALL _libusb_kernel_driver_active(libusb_device_handle *dev_handle, int interface_number) {
    return LIBUSB_SUCCESS;
}

int LIBUSB_CALL _libusb_cancel_transfer(struct libusb_transfer *transfer) {
    transfer->status = LIBUSB_TRANSFER_CANCELLED;
    return LIBUSB_SUCCESS;
}

void LIBUSB_CALL _libusb_free_transfer(struct libusb_transfer *transfer) {
    free(transfer);
}

int LIBUSB_CALL _libusb_set_interface_alt_setting(libusb_device_handle *dev_handle,
	int interface_number, int alternate_setting) {
    val device = val::global("device");

    if (!device.as<bool>())
        return LIBUSB_ERROR_NO_DEVICE;

    device.call<val>("selectAlternateInterface", interface_number, alternate_setting).await();

    return LIBUSB_SUCCESS;
}

int LIBUSB_CALL _libusb_handle_events_timeout(libusb_context *ctx, struct timeval *tv) {
    return libusb_handle_events_timeout_completed(ctx, tv, nullptr);
}

int LIBUSB_CALL _libusb_handle_events_timeout_completed(libusb_context *ctx,
	struct timeval *tv, int *completed) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        std::move(staging.begin(), staging.end(), std::back_inserter(transfers));
        staging.clear();
    }

    transfers.erase(
    std::remove_if(transfers.begin(), transfers.end(),
        [](struct libusb_transfer* transfer) {
            if (transfer->status == LIBUSB_TRANSFER_COMPLETED ||
                transfer->status == LIBUSB_TRANSFER_CANCELLED) {
                transfer->callback(transfer);
                return true;
            }
            return false;
        }),
    transfers.end());

    return LIBUSB_SUCCESS;
}