#include <thread>
#include <chrono>

#include "jetstream/base.hh"

using namespace Jetstream;

void PrintVarDebug(const std::string& varName, const auto& a) {
    JST_DEBUG("{} | data = {}, size = {}, refs = {}, ptr = {}", 
             varName, a.data() != nullptr, a.size(), a.refs(), 
             fmt::ptr(a.data()));
}

int main() {
    // Initialize the backends.
    if (Backend::Initialize<Device::CPU>({}) != Result::SUCCESS) {
        JST_ERROR("Cannot initialize CPU backend.");
        return 1;
    }

    if (Backend::Initialize<Device::Metal>({}) != Result::SUCCESS) {
        JST_ERROR("Cannot initialize Metal backend.");
        return 1;
    }

    {
        Vector<Device::CPU, U64, 4> a({1, 2, 3, 4});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        Vector<Device::CPU, U64, 4> c(a);
        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.refs() == 2);
        assert(c.data() != nullptr);

        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 2);
        assert(a.data() != nullptr);
    }

    JST_TRACE("====")


    {
        Vector<Device::CPU, U64> a;
        PrintVarDebug("a", a);
        assert(a.size() == 0);
        assert(a.refs() == 0);
        assert(a.data() == nullptr);

        JST_INFO("Vector empty creation test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Vector<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        JST_INFO("Vector filled creation test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Vector<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        Vector<Device::CPU, U64> c(a);
        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.refs() == 2);
        assert(c.data() != nullptr);

        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 2);
        assert(a.data() != nullptr);

        JST_INFO("Vector copy constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        {
            
        }
        Vector<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        Vector<Device::CPU, U64> c(std::move(a));
        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.refs() == 2);
        assert(c.data() != nullptr);

        JST_INFO("Vector move constructor test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Vector<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        Vector<Device::CPU, U64> c; 
        PrintVarDebug("c", c);
        assert(c.size() == 0);
        assert(c.refs() == 0);
        assert(c.data() == nullptr);

        c = a; 

        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 2);
        assert(a.data() != nullptr);

        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.refs() == 2);
        assert(c.data() != nullptr);

        JST_INFO("Vector copy test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Vector<Device::CPU, U64> a({24});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        Vector<Device::CPU, U64> c; 
        PrintVarDebug("c", c);
        assert(c.size() == 0);
        assert(c.refs() == 0);
        assert(c.data() == nullptr);

        c = std::move(a); 

        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 2);
        assert(a.data() != nullptr);

        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.refs() == 2);
        assert(c.data() != nullptr);

        JST_INFO("Vector move test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        struct Test {
            Vector<Device::CPU, U64> pickles;
        };

        Test test = Test{
            .pickles = Vector<Device::CPU, U64>({24}),
        };

        PrintVarDebug("test.pickles", test.pickles);
        assert(test.pickles.size() == 24);
        assert(test.pickles.refs() == 1);
        assert(test.pickles.data() != nullptr);

        JST_INFO("Vector struct test successful!");
    }

    JST_INFO("---------------------------------------------");

    {
        Vector<Device::CPU, F32> b({42});
        {
            Vector<Device::Metal, F32> a({3});
            b = a;
            JST_TRACE("{}", a.hash());
        }

        JST_TRACE("{}", b.hash());
    }

    JST_INFO("---------------------------------------------");

    {
        Vector<Device::CPU, F32> b({42});
        Vector<Device::Metal, F32> a(b);
    }

    JST_INFO("---------------------------------------------");

    {
        Vector<Device::CPU, F32> b(nullptr, {21});
    }

    JST_INFO("---------------------------------------------");

    {
        Vector<Device::CPU, F32> b({43});

        {
            Vector<Device::Metal, F32> a(b);
            JST_TRACE("{}", a.hash());
        }

        JST_TRACE("{}", b.hash());
    }

    return 0;
}
