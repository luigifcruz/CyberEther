#include <thread>
#include <chrono>

#include "jetstream/memory/base.hh"

// TODO: Use Catch2 to implement proper unit tests.
// TODO: Drastically improve test coverage.

using namespace Jetstream;

// Print var debug info.

template<Device D, typename T>
void PrintVarDebug(const std::string& varName, const Tensor<D, T>& a) {
    if constexpr (D == Device::CPU) {
        JST_DEBUG("{} | [C] empty = {}, size = {}, references = {}, ptr = {}", 
             varName, a.empty(), a.size(), a.references(), 
             jst::fmt::ptr(a.data()));
    }

    if constexpr (D == Device::Metal) {
        JST_DEBUG("{} | [M] empty = {}, size = {}, references = {}, ptr = {}", 
             varName, a.empty(), a.size(), a.references(), jst::fmt::ptr(a.data()));
    }
}

int main() {
    {
        Tensor<Device::CPU, F32> a;

        a.set_locale({"a", "b", "c"});
        assert(a.locale().blockId == "a");
        assert(a.locale().moduleId == "b");
        assert(a.locale().pinId == "c");

        JST_INFO("Tensor locale test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> a;

        a.set_locale({"a", "b", "c"});
        assert(a.locale().blockId == "a");
        assert(a.locale().moduleId == "b");
        assert(a.locale().pinId == "c");

        Tensor<Device::CPU, F32> c(a);
        assert(c.locale().blockId == "a");
        assert(c.locale().moduleId == "b");
        assert(c.locale().pinId == "c");

        JST_INFO("Tensor locale copy constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> a;

        a.set_locale({"a", "b", "c"});
        assert(a.locale().blockId == "a");
        assert(a.locale().moduleId == "b");
        assert(a.locale().pinId == "c");

        Tensor<Device::CPU, F32> c;
        assert(c.locale().empty() == true);

        a = c;
        assert(a.locale().blockId == "a");
        assert(a.locale().moduleId == "b");
        assert(a.locale().pinId == "c");

        JST_INFO("Tensor locale copy test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, U64> a({1, 2, 3, 4});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.references() == 1);
        assert(a.data() != nullptr);

        Tensor<Device::CPU, U64> c(a);
        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.references() == 2);
        assert(c.data() != nullptr);

        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.references() == 2);
        assert(a.data() != nullptr);

        JST_INFO("Tensor copy constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, U64> a;
        PrintVarDebug("a", a);
        assert(a.size() == 0);
        assert(a.references() == 1);
        assert(a.data() == nullptr);

        JST_INFO("Vector empty creation test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.references() == 1);
        assert(a.data() != nullptr);

        JST_INFO("Vector filled creation test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.references() == 1);
        assert(a.data() != nullptr);

        Tensor<Device::CPU, U64> c(a);
        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.references() == 2);
        assert(c.data() != nullptr);

        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.references() == 2);
        assert(a.data() != nullptr);

        JST_INFO("Vector copy constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.references() == 1);
        assert(a.data() != nullptr);

        Tensor<Device::CPU, U64> c(std::move(a));
        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.references() == 1);
        assert(c.data() != nullptr);

        JST_INFO("Vector move constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.references() == 1);
        assert(a.data() != nullptr);

        Tensor<Device::CPU, U64> c; 
        PrintVarDebug("c", c);
        assert(c.size() == 0);
        assert(c.references() == 1);
        assert(c.data() == nullptr);

        c = a; 

        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.references() == 2);
        assert(a.data() != nullptr);

        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.references() == 2);
        assert(c.data() != nullptr);

        JST_INFO("Vector copy test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.references() == 1);
        assert(a.data() != nullptr);

        Tensor<Device::CPU, U64> c; 
        PrintVarDebug("c", c);
        assert(c.size() == 0);
        assert(c.references() == 1);
        assert(c.data() == nullptr);

        c = std::move(a); 

        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.references() == 1);
        assert(c.data() != nullptr);

        JST_INFO("Vector move test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        struct Test {
            Tensor<Device::CPU, U64> pickles;
        };

        Test test = Test{
            .pickles = Tensor<Device::CPU, U64>({24}),
        };

        PrintVarDebug("test.pickles", test.pickles);
        assert(test.pickles.size() == 24);
        assert(test.pickles.references() == 1);
        assert(test.pickles.data() != nullptr);

        JST_INFO("Vector struct test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> b(nullptr, {21});
        PrintVarDebug("b", b);
        assert(b.size() == 21);
        assert(b.references() == 1);
        assert(b.data() == nullptr);

        JST_INFO("Vector raw pointer test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> array;
        PrintVarDebug("array", array);

        JST_INFO("Tensor default constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> array({1, 2, 3});
        PrintVarDebug("array", array);

        JST_INFO("Tensor shape constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> array({2, 3, 4});
        PrintVarDebug("array", array);

        assert(array.size() == 2 * 3 * 4);
        assert(array.size_bytes() == 2 * 3 * 4 * sizeof(F32));
        assert(array.element_size() == sizeof(F32));
        assert(array.rank() == 3);
        assert(array.ndims() == 3);
        assert(array.shape(1) == 3);
        assert(array.stride(1) == 4);

        JST_INFO("Tensor properties test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> array({2, 3, 4});
        PrintVarDebug("array", array);

        array.attribute("key").set(std::string("value"));
        assert(array.attribute("key").template get<std::string>() == "value");

        JST_INFO("Tensor metadata test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> array({2, 3, 4});
        PrintVarDebug("array", array);

        F32* data = array.data();
        assert(data != nullptr);

        JST_INFO("Tensor storage test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> cpu_array(nullptr, {2, 3, 4});
        PrintVarDebug("cpu_array", cpu_array);

        assert(cpu_array.shape(0) == 2);
        assert(cpu_array.shape(1) == 3);
        assert(cpu_array.shape(2) == 4);

        JST_INFO("Tensor cloning with data and shape test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> array({2, 4});
        PrintVarDebug("array", array);

        array[{0, 0}] = 1;
        array[{0, 1}] = 2;
        array[{0, 2}] = 3;
        array[{0, 3}] = 4;
        array[{1, 0}] = 5;
        array[{1, 1}] = 6;
        array[{1, 2}] = 7;
        array[{1, 3}] = 8;
        assert((array[{0, 0}] == 1));
        assert((array[{0, 1}] == 2));
        assert((array[{0, 2}] == 3));
        assert((array[{0, 3}] == 4));
        assert((array[{1, 0}] == 5));
        assert((array[{1, 1}] == 6));
        assert((array[{1, 2}] == 7));
        assert((array[{1, 3}] == 8));

        JST_INFO("Tensor operator[] with two dimensions test successful!");
    } 

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> array({2, 2, 2});
        PrintVarDebug("array", array);

        array[{0, 0, 0}] = 1;
        array[{0, 0, 1}] = 2;
        array[{0, 1, 0}] = 3;
        array[{0, 1, 1}] = 4;
        array[{1, 0, 0}] = 5;
        array[{1, 0, 1}] = 6;
        array[{1, 1, 0}] = 7;
        array[{1, 1, 1}] = 8;
        assert((array[{0, 0, 0}] == 1));
        assert((array[{0, 0, 1}] == 2));
        assert((array[{0, 1, 0}] == 3));
        assert((array[{0, 1, 1}] == 4));
        assert((array[{1, 0, 0}] == 5));
        assert((array[{1, 0, 1}] == 6));
        assert((array[{1, 1, 0}] == 7));
        assert((array[{1, 1, 1}] == 8));

        JST_INFO("Tensor operator[] with three dimensions test successful!");
    } 

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> og_array({2, 3, 4});
        PrintVarDebug("og_array", og_array);

        og_array[{0, 0, 0}] = 1;
        og_array[{0, 0, 1}] = 2;
        og_array[{0, 0, 2}] = 3;
        og_array[{0, 0, 3}] = 4;
        og_array[{0, 1, 0}] = 5;
        og_array[{0, 1, 1}] = 6;
        og_array[{0, 1, 2}] = 7;
        og_array[{0, 1, 3}] = 8;
        og_array[{0, 2, 0}] = 9;
        og_array[{0, 2, 1}] = 10;
        og_array[{0, 2, 2}] = 11;
        og_array[{0, 2, 3}] = 12;
        og_array[{1, 0, 0}] = 13;
        og_array[{1, 0, 1}] = 14;
        og_array[{1, 0, 2}] = 15;
        og_array[{1, 0, 3}] = 16;
        og_array[{1, 1, 0}] = 17;
        og_array[{1, 1, 1}] = 18;
        og_array[{1, 1, 2}] = 19;
        og_array[{1, 1, 3}] = 20;
        og_array[{1, 2, 0}] = 21;
        og_array[{1, 2, 1}] = 22;
        og_array[{1, 2, 2}] = 23;
        og_array[{1, 2, 3}] = 24;

        og_array.slice({0, {}, 0});  // og_array[0, :, 0]
        PrintVarDebug("og_array", og_array);

        assert(og_array.rank() == 1);
        assert(og_array.shape(0) == 3);

        JST_DEBUG("{}", og_array[std::vector<U64>{0}]);
        JST_DEBUG("{}", og_array[std::vector<U64>{1}]);
        JST_DEBUG("{}", og_array[std::vector<U64>{2}]);

        assert((og_array[std::vector<U64>{0}] == 1));
        assert((og_array[std::vector<U64>{1}] == 5));
        assert((og_array[std::vector<U64>{2}] == 9));

        JST_INFO("Tensor test successful!");
    }

    JST_INFO("---------------------------------------------");


    {
        Tensor<Device::CPU, F32> array({64});
        PrintVarDebug("array", array);

        array.slice({{0, 0, 2}});

        assert(array.shape() == std::vector<U64>{32});
        assert(array.stride() == std::vector<U64>{2});
        assert(array.size() == 32);
        assert(array.contiguous() == false);

        JST_INFO("Tensor test successful!");
    }

    JST_INFO("---------------------------------------------");

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    {
        Tensor<Device::CPU, F32> b({42});
        PrintVarDebug("b", b);
        assert(b.size() == 42);
        assert(b.references() == 1);
        assert(b.data() != nullptr);

        {
            Tensor<Device::Metal, F32> a({3});
            PrintVarDebug("a", a);
            assert(a.size() == 3);
            assert(a.references() == 1);

            b = a;
            PrintVarDebug("b", b);
            assert(b.size() == 3);
            assert(b.references() == 2);
            assert(b.data() != nullptr);
        }
        PrintVarDebug("b", b);
        assert(b.size() == 3);
        assert(b.references() == 1);
        assert(b.data() != nullptr);

        JST_INFO("Vector cross-device test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> b({42});
        PrintVarDebug("b", b);
        assert(b.size() == 42);
        assert(b.references() == 1);

        Tensor<Device::Metal, F32> a(b);
        PrintVarDebug("a", a);
        assert(a.size() == 42);
        assert(a.references() == 2);

        JST_INFO("Vector cross-device copy constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::Metal, F32> metal_array({1, 2, 3});
        PrintVarDebug("metal_array", metal_array);
    
        Tensor<Device::CPU, F32> array(metal_array);
        PrintVarDebug("array", array);

        JST_INFO("Tensor cross-device constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::Metal, F32> metal_array({1, 2, 3});
        PrintVarDebug("metal_array", metal_array);

        metal_array.attribute("test").set(1);

        Tensor<Device::CPU, F32> array;
        PrintVarDebug("array", array);

        array = metal_array;

        assert(array.attributes().contains("test") == true);
        assert(array.attributes().contains("test2") == false);

        JST_INFO("Tensor store after copy test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::CPU, F32> cpu_array;
        PrintVarDebug("cpu_array", cpu_array);
        assert(cpu_array.device() == Device::CPU);

        Tensor<Device::Metal, F32> metal_array;
        PrintVarDebug("metal_array", metal_array);
        assert(metal_array.device() == Device::Metal);

        JST_INFO("Tensor device indicator test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::Metal, F32> metal_array({2, 3, 4});
        PrintVarDebug("metal_array", metal_array);
        assert(metal_array.compatible_devices().contains(Device::CPU));

        Tensor<Device::CPU, F32> cpu_array(metal_array);
        PrintVarDebug("cpu_array", cpu_array);
        assert(cpu_array.compatible_devices().contains(Device::Metal));

        JST_INFO("Tensor compatibility test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Tensor<Device::Metal, F32> metal_array({2, 3, 4});
        PrintVarDebug("metal_array", metal_array);
        Tensor<Device::CPU, F32> cpu_array;
        PrintVarDebug("cpu_array", cpu_array);
        cpu_array = metal_array;
        PrintVarDebug("cpu_array", cpu_array);
        assert(cpu_array.shape(0) == 2);
        assert(cpu_array.shape(1) == 3);
        assert(cpu_array.shape(2) == 4);

        JST_INFO("Tensor cloning test successful!");
    }

    JST_INFO("---------------------------------------------");
#endif

#if defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE) && \
    defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    {
        Tensor<Device::CUDA, F32> b({42});
        PrintVarDebug("b", b);
        assert(b.size() == 42);
        assert(b.references() == 1);
        assert(b.data() != nullptr);

        {
            Tensor<Device::Vulkan, F32> a({3});
            PrintVarDebug("a", a);
            assert(a.size() == 3);
            assert(a.references() == 1);

            b = a;
            PrintVarDebug("b", b);
            assert(b.size() == 3);
            assert(b.references() == 2);
            assert(b.data() != nullptr);
        }
        PrintVarDebug("b", b);
        assert(b.size() == 3);
        assert(b.references() == 1);
        assert(b.data() != nullptr);

        JST_INFO("Vector cross-device test successful!");
    }
#endif

    JST_INFO("Test successful!");

    return 0;
}
