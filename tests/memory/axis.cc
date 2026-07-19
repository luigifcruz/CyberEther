#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "jetstream/memory/axis.hh"

using namespace Jetstream;

TEST_CASE("ResolveAxis normalizes regular tensor axes", "[memory][axis]") {
    REQUIRE(ResolveAxis(0, 3) == Index{0});
    REQUIRE(ResolveAxis(2, 3) == Index{2});
    REQUIRE(ResolveAxis(-1, 3) == Index{2});
    REQUIRE(ResolveAxis(-3, 3) == Index{0});

    REQUIRE_FALSE(ResolveAxis(0, 0));
    REQUIRE_FALSE(ResolveAxis(3, 3));
    REQUIRE_FALSE(ResolveAxis(-4, 3));
    REQUIRE_FALSE(ResolveAxis(std::numeric_limits<I64>::min(), 3));
    REQUIRE_FALSE(ResolveAxis(std::numeric_limits<I64>::max(), 3));
    REQUIRE_FALSE(ResolveAxis(0, std::numeric_limits<Index>::max()));
}

TEST_CASE("ResolveInsertionAxis normalizes dimension insertion axes",
          "[memory][axis]") {
    REQUIRE(ResolveInsertionAxis(0, 2) == Index{0});
    REQUIRE(ResolveInsertionAxis(2, 2) == Index{2});
    REQUIRE(ResolveInsertionAxis(-1, 2) == Index{2});
    REQUIRE(ResolveInsertionAxis(-2, 2) == Index{1});
    REQUIRE(ResolveInsertionAxis(-3, 2) == Index{0});

    REQUIRE(ResolveInsertionAxis(0, 0) == Index{0});
    REQUIRE(ResolveInsertionAxis(-1, 0) == Index{0});
    REQUIRE_FALSE(ResolveInsertionAxis(3, 2));
    REQUIRE_FALSE(ResolveInsertionAxis(-4, 2));
    REQUIRE_FALSE(ResolveInsertionAxis(std::numeric_limits<I64>::min(), 2));
    REQUIRE_FALSE(ResolveInsertionAxis(std::numeric_limits<I64>::max(), 2));
    REQUIRE_FALSE(ResolveInsertionAxis(0, std::numeric_limits<Index>::max()));
}

int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
