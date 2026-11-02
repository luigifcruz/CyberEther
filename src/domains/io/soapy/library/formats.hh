#ifndef JETSTREAM_DOMAINS_IO_SOAPY_LIBRARY_FORMATS_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_LIBRARY_FORMATS_HH

#include <array>

#include <jetstream/memory/types.hh>

namespace Jetstream::Modules {

struct SoapySampleFormat {
    DataType dtype;
    const char* name;
    F32 fullScale;
};

inline const std::array<SoapySampleFormat, 5> SoapySampleFormats{{
    {DataType::CF32, "CF32", 1.0f},
    {DataType::CF64, "CF64", 1.0f},
    {DataType::CI16, "CS16", 32768.0f},
    {DataType::CI8, "CS8", 128.0f},
    {DataType::CU8, "CU8", 128.0f},
}};

inline const SoapySampleFormat* FindSoapySampleFormat(const DataType& dtype) {
    for (const auto& format : SoapySampleFormats) {
        if (format.dtype == dtype) {
            return &format;
        }
    }
    return nullptr;
}

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_LIBRARY_FORMATS_HH
