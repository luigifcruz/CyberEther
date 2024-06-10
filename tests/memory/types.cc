#include <sstream>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "jetstream/memory/types.hh"

using namespace Jetstream;

TEST_CASE("Locale Struct Tests", "[Locale]") {
    SECTION("Default Constructor and Empty Check") {
        Locale locale;
        REQUIRE(locale.empty() == true);
    }

    SECTION("Parent Block, Block, Module, and Pin Methods") {
        Locale locale{"block1", "module1", "pin1"};

        REQUIRE(locale.block() == Locale{"block1"});
        REQUIRE(locale.module() == Locale{"block1", "module1"});
        REQUIRE(locale.pin() == Locale{"block1", "", "pin1"});
    }

    SECTION("Equality Operator") {
        Locale locale1{"block1", "module1", "pin1"};
        Locale locale2{"block1", "module1", "pin1"};
        Locale locale3{"block2", "module2", "pin2"};

        REQUIRE(locale1 == locale2);
        REQUIRE_FALSE(locale1 == locale3);
    }

    SECTION("String Hash Method") {
        Locale locale{"block1", "module1", "pin1"};
        REQUIRE(locale.shash() == "block1module1pin1");
    }

    SECTION("Hash Method") {
        Locale locale{"block1", "module1", "pin1"};
        REQUIRE(locale.hash() == Locale::Hasher()(locale));
    }

    SECTION("Output Stream Overload") {
        Locale locale{"block1", "module1", "pin1"};
        std::ostringstream os;
        os << locale;
        REQUIRE(os.str() == "block1-module1.pin1");
    }

    SECTION("Partial and Empty Fields") {
        Locale onlyBlock{"block1"};
        Locale onlyModule{"", "module1"};
        Locale onlyPin{"", "", "pin1"};
        Locale blockAndModule{"block1", "module1"};
        Locale blockAndPin{"block1", "", "pin1"};
        Locale moduleAndPin{"", "module1", "pin1"};
        Locale emptyLocale;

        REQUIRE(onlyBlock.empty() == false);
        REQUIRE(onlyModule.empty() == false);
        REQUIRE(onlyPin.empty() == false);
        REQUIRE(blockAndModule.empty() == false);
        REQUIRE(blockAndPin.empty() == false);
        REQUIRE(moduleAndPin.empty() == false);
        REQUIRE(emptyLocale.empty() == true);

        REQUIRE(onlyBlock.shash() == "block1");
        REQUIRE(onlyModule.shash() == "module1");
        REQUIRE(onlyPin.shash() == "pin1");
        REQUIRE(blockAndModule.shash() == "block1module1");
        REQUIRE(blockAndPin.shash() == "block1pin1");
        REQUIRE(moduleAndPin.shash() == "module1pin1");

        std::ostringstream os;
        os << onlyBlock;
        REQUIRE(os.str() == "block1");

        os.str("");
        os << onlyModule;
        REQUIRE(os.str() == "module1");

        os.str("");
        os << onlyPin;
        REQUIRE(os.str() == ".pin1");

        os.str("");
        os << blockAndModule;
        REQUIRE(os.str() == "block1-module1");

        os.str("");
        os << blockAndPin;
        REQUIRE(os.str() == "block1.pin1");

        os.str("");
        os << moduleAndPin;
        REQUIRE(os.str() == "module1.pin1");

        os.str("");
        os << emptyLocale;
        REQUIRE(os.str() == "");
    }

    SECTION("Locale Kind Identifier") {
        SECTION("Block") {
            Locale locale{"block1"};
            REQUIRE(locale.isBlock());
            REQUIRE_FALSE(locale.isModule());
            REQUIRE_FALSE(locale.isPin());
        }

        SECTION("Module") {
            Locale locale{"block1", "module1"};
            REQUIRE_FALSE(locale.isBlock());
            REQUIRE(locale.isModule());
            REQUIRE_FALSE(locale.isPin());
        }

        SECTION("Module inside Internal Block") {
            Locale locale{"block1", "module1"};
            REQUIRE_FALSE(locale.isBlock());
            REQUIRE(locale.isModule());
            REQUIRE_FALSE(locale.isPin());
        }

        SECTION("Pin") {
            Locale locale{"block1", "module1", "pin1"};
            REQUIRE_FALSE(locale.isBlock());
            REQUIRE_FALSE(locale.isModule());
            REQUIRE(locale.isPin());
        }
    }

    SECTION("Hash Consistency and Uniqueness") {
        Locale locale1{"block1", "module1", "pin1"};
        Locale locale2{"block1", "module2", "pin1"};
        Locale locale3{"block1", "module1", "pin2"};

        REQUIRE(locale1.hash() != locale2.hash());
        REQUIRE(locale1.hash() != locale3.hash());
        REQUIRE(locale2.hash() != locale3.hash());
    }
}

TEST_CASE("Range Struct Tests", "[Range]") {
    SECTION("Operator== for Floating Point") {
        Range<float> range1{10.0f, 20.0f};
        Range<float> range2{10.0f, 20.0f};
        Range<float> range3{15.0f, 25.0f};

        REQUIRE(range1 == range2);
        REQUIRE_FALSE(range1 == range3);
    }

    SECTION("Operator!= for Floating Point") {
        Range<float> range1{10.0f, 20.0f};
        Range<float> range2{15.0f, 25.0f};

        REQUIRE(range1 != range2);
    }

    SECTION("Operator<= for Floating Point") {
        Range<float> range1{10.0f, 20.0f};
        Range<float> range2{15.0f, 25.0f};
        Range<float> range3{5.0f, 15.0f};

        REQUIRE(range1 <= range2);
        REQUIRE_FALSE(range2 <= range1);
        REQUIRE(range3 <= range1);
    }
}

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(4);

    return Catch::Session().run(argc, argv);
}