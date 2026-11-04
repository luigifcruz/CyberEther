#ifndef JETSTREAM_RENDER_METAL_PROGRAM_CACHE_HH
#define JETSTREAM_RENDER_METAL_PROGRAM_CACHE_HH

#include <algorithm>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

#include "jetstream/backend/devices/metal/base.hh"

namespace Jetstream::Render::Detail {

struct MetalProgramCache {
    static constexpr U64 MaxDevices = 4;
    static constexpr U64 MaxFunctions = 64;
    static constexpr U64 MaxPipelines = 128;

    struct Function {
        NS::SharedPtr<MTL::Function> handle;
        U64 used;
    };

    struct Pipeline {
        NS::SharedPtr<MTL::Function> vertex;
        NS::SharedPtr<MTL::Function> fragment;
        NS::SharedPtr<MTL::RenderPipelineState> handle;
        U64 used;
    };

    struct Device {
        NS::SharedPtr<MTL::Device> handle;
        std::unordered_map<std::string, Function> functions;
        std::unordered_map<std::string, Pipeline> pipelines;
        U64 clock = 0;

        explicit Device(MTL::Device* device) : handle(NS::RetainPtr(device)) {}

        NS::SharedPtr<MTL::Function> findFunction(const std::string& key) {
            if (const auto it = functions.find(key); it != functions.end()) {
                it->second.used = ++clock;
                return it->second.handle;
            }
            return {};
        }

        void insertFunction(std::string key, NS::SharedPtr<MTL::Function> function) {
            functions.emplace(std::move(key), Function{std::move(function), ++clock});
            trim(functions, MaxFunctions);
        }

        NS::SharedPtr<MTL::RenderPipelineState> findPipeline(const std::string& key) {
            if (const auto it = pipelines.find(key); it != pipelines.end()) {
                it->second.used = ++clock;
                return it->second.handle;
            }
            return {};
        }

        void insertPipeline(std::string key,
                            NS::SharedPtr<MTL::Function> vertex,
                            NS::SharedPtr<MTL::Function> fragment,
                            NS::SharedPtr<MTL::RenderPipelineState> pipeline) {
            pipelines.emplace(std::move(key), Pipeline{
                std::move(vertex), std::move(fragment), std::move(pipeline), ++clock,
            });
            trim(pipelines, MaxPipelines);
        }

     private:
        template<typename Entries>
        static void trim(Entries& entries, U64 capacity) {
            if (entries.size() > capacity) {
                const auto oldest = std::min_element(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
                    return a.second.used < b.second.used;
                });
                entries.erase(oldest);
            }
        }
    };

    std::mutex mutex;
    std::list<Device> devices;

    Device& forDevice(MTL::Device* device) {
        const auto it = std::find_if(devices.begin(), devices.end(), [device](const auto& entry) {
            return entry.handle.get() == device;
        });
        if (it != devices.end()) {
            devices.splice(devices.begin(), devices, it);
        } else {
            devices.emplace_front(device);
            if (devices.size() > MaxDevices) {
                devices.pop_back();
            }
        }
        return devices.front();
    }
};

}  // namespace Jetstream::Render::Detail

#endif
