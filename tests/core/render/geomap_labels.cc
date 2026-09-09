#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>

#include "render/components/geomap_labels.hh"

using namespace Jetstream;
using namespace Jetstream::Render::Components;

namespace {

struct Candidate {
    GeoMapLabels::Kind kind;
    U8 flags;
    I32 scalerank;
    I32 population;
    F32 minZoom;
    std::string name;
    F32 lon;
    F32 lat;
    F32 area = 0.0f;
    std::vector<Extent2D<F32>> path;
};

}  // namespace

TEST_CASE("GeoMap renderer uses the tested tuning-aware label policy", "[render][geomap]") {
    GeoMapLabels::Tuning tuning;
    auto full = [&](const Candidate& candidate) {
        return GeoMapLabels::FullVisibilityZoom(candidate, tuning, 0.0f, 0.0f);
    };
    Candidate country{GeoMapLabels::Kind::Country, 0, 2, 0, 1.7f, "Country", 0, 0};
    REQUIRE(full(country) == -1.0f);
    tuning.countryMajorFull = 0.5f;
    REQUIRE(full(country) == 0.5f);
    country.minZoom = 5.0f;
    REQUIRE(full(country) == Catch::Approx(2.3f));

    Candidate state{GeoMapLabels::Kind::State, 0, 2, 0, 18.0f, "State", 0, 0, 55};
    REQUIRE(full(state) == 1.75f); // Area/rank, not the flat-map source threshold.
    tuning.stateAreaMaxLead = 0.5f;
    REQUIRE(full(state) == 2.25f);
    tuning.stateMajorFull = 3;
    REQUIRE(full(state) == 2.5f);

    Candidate city{GeoMapLabels::Kind::City, 0, 10, 1000000, 9, "City", 0, 0};
    REQUIRE(full(city) == 6); // max(source minus lead, population tier).
    tuning.citySourceLead = 5;
    REQUIRE(full(city) == 4.5f);
    tuning.cityPopulationFull[1] = 7;
    REQUIRE(full(city) == 7);
    city.flags = 2;
    tuning.worldCityFull = 1.5f;
    REQUIRE(full(city) == 1.5f); // World cities bypass source thresholds.

    Candidate capital{GeoMapLabels::Kind::Capital, 8, 0, 0, 0, "Capital", 0, 0};
    tuning.nationalCapitalFull = 3;
    REQUIRE(full(capital) == 3);
    Candidate airport{GeoMapLabels::Kind::Airport, 1, 10, 0, 20, "APT", 0, 0};
    tuning.airportFull[0] = 4;
    REQUIRE(full(airport) == 4);

    Candidate region{GeoMapLabels::Kind::Region, 0, 0, 0, 1, "Range", 0, 0};
    region.path = {{0, 0}, {1, 1}};
    REQUIRE(full(region) == 1.5f);
    tuning.pathLabelFull = 2;
    REQUIRE(full(region) == 2);
    region.path.clear();
    REQUIRE(GeoMapLabels::FullVisibilityZoom(region, tuning, 3, 0.5f) == 3.5f);
}

TEST_CASE("GeoMap ordinary and path labels share fade-in and fade-out rules", "[render][geomap]") {
    REQUIRE(GeoMapLabels::Visibility(3, 4, 2, 1) == 0);
    REQUIRE(GeoMapLabels::Visibility(3.5f, 4, 2, 1) == Catch::Approx(0.5f));
    REQUIRE(GeoMapLabels::Visibility(4, 4, 2, 1) == 1);
    REQUIRE(GeoMapLabels::Visibility(4.5f, 4, 2, 1) == Catch::Approx(0.5f));
    REQUIRE(GeoMapLabels::Visibility(5, 4, 2, 1) == 0);
    REQUIRE(GeoMapLabels::Visibility(20, 4, 0, 1) == 1);
}

TEST_CASE("GeoMap label hierarchy prioritizes capitals and world cities",
          "[render][geomap]") {
    const Candidate national{
        GeoMapLabels::Kind::Capital, 8, 4, 100000, 0.0f, "Capital", 0, 0};
    const Candidate regional{
        GeoMapLabels::Kind::Capital, 1, 0, 10000000, 0.0f, "Regional", 0, 0};
    REQUIRE(GeoMapLabels::MoreImportant(national, regional));

    const Candidate worldCity{
        GeoMapLabels::Kind::City, 2, 5, 100000, 0.0f, "World", 0, 0};
    const Candidate ordinaryCity{
        GeoMapLabels::Kind::City, 0, 0, 10000000, 0.0f, "Ordinary", 0, 0};
    REQUIRE(GeoMapLabels::MoreImportant(worldCity, ordinaryCity));
}

TEST_CASE("GeoMap label visibility fades before the source threshold",
          "[render][geomap]") {
    const F32 fullZoom = GeoMapLabels::FullVisibilityZoom(2.1f, 0, 0.0f);
    REQUIRE(fullZoom == 2.1f);
    REQUIRE(GeoMapLabels::VisibilityFade(0.1f, fullZoom, 2.0f) == 0.0f);
    REQUIRE(GeoMapLabels::VisibilityFade(1.1f, fullZoom, 2.0f) > 0.0f);
    REQUIRE(GeoMapLabels::VisibilityFade(2.1f, fullZoom, 2.0f) == 1.0f);
    REQUIRE(GeoMapLabels::VisibilityFadeOut(4.0f, 6.0f, 2.0f) == 1.0f);
    REQUIRE(GeoMapLabels::VisibilityFadeOut(5.0f, 6.0f, 2.0f) > 0.0f);
    REQUIRE(GeoMapLabels::VisibilityFadeOut(6.0f, 6.0f, 2.0f) == 0.0f);

    REQUIRE(GeoMapLabels::FullVisibilityZoom(1.7f, 0, 3.0f) == 3.0f);
}

TEST_CASE("GeoMap overview prioritizes countries before state detail",
           "[render][geomap]") {
    const F32 usaFullZoom = GeoMapLabels::CountryFullVisibilityZoom(2, 1.7f);
    const F32 usaFadeRange = GeoMapLabels::CountryFadeRange(2);
    REQUIRE(usaFullZoom == -1.0f);
    REQUIRE(GeoMapLabels::VisibilityFade(usaFullZoom, usaFullZoom,
                                         usaFadeRange) == 1.0f);
    const F32 stateLineRange = GeoMapLabels::StateLineFullZoom -
                               GeoMapLabels::StateLineFadeStartZoom;
    REQUIRE(usaFullZoom < GeoMapLabels::StateLineFadeStartZoom);
    REQUIRE(GeoMapLabels::VisibilityFade(
                GeoMapLabels::StateLineFadeStartZoom,
                GeoMapLabels::StateLineFullZoom,
                stateLineRange) == 0.0f);
    REQUIRE(GeoMapLabels::VisibilityFade(3.0f,
                                         GeoMapLabels::StateLineFullZoom,
                                         stateLineRange) > 0.0f);

    const F32 stateLabelRange = GeoMapLabels::StateLabelFullZoom -
                                GeoMapLabels::StateLabelFadeStartZoom;
    REQUIRE(GeoMapLabels::VisibilityFade(
                GeoMapLabels::StateLabelFadeStartZoom,
                GeoMapLabels::StateLabelFullZoom,
                stateLabelRange) == 0.0f);
    REQUIRE(GeoMapLabels::VisibilityFade(3.25f,
                                         GeoMapLabels::StateLabelFullZoom,
                                         stateLabelRange) > 0.0f);

    REQUIRE(GeoMapLabels::RiverFullVisibilityZoom(0) == 0.0f);
    REQUIRE(GeoMapLabels::RiverFullVisibilityZoom(3) == 0.0f);
    REQUIRE(GeoMapLabels::RiverFullVisibilityZoom(4) == 1.0f);
    REQUIRE(GeoMapLabels::RiverFullVisibilityZoom(10) == 7.0f);
    REQUIRE(GeoMapLabels::RiverLineWidth(0) == 5.0f);
    REQUIRE(GeoMapLabels::RiverLineWidth(3) >
            GeoMapLabels::RiverLineWidth(4));
    REQUIRE(GeoMapLabels::RiverLineWidth(10) == 1.5f);

    REQUIRE(GeoMapLabels::CityTierFullVisibilityZoom(
                GeoMapLabels::Kind::City, 2, 1000) == 1.0f);
    REQUIRE(GeoMapLabels::CityTierFullVisibilityZoom(
                GeoMapLabels::Kind::Capital, 9, 1000) == 2.0f);
    REQUIRE(GeoMapLabels::CityTierFullVisibilityZoom(
                GeoMapLabels::Kind::City, 4, 1000) == 3.5f);
    REQUIRE(GeoMapLabels::CityTierFullVisibilityZoom(
                GeoMapLabels::Kind::City, 0, 5000000) == 3.5f);
    REQUIRE(GeoMapLabels::CityTierFullVisibilityZoom(
                GeoMapLabels::Kind::City, 0, 1000000) == 4.5f);
    REQUIRE(GeoMapLabels::CityTierFullVisibilityZoom(
                GeoMapLabels::Kind::City, 0, 250000) == 5.0f);
    REQUIRE(GeoMapLabels::CityTierFullVisibilityZoom(
                GeoMapLabels::Kind::City, 0, 50000) == 6.5f);
    REQUIRE(GeoMapLabels::CityTierFullVisibilityZoom(
                GeoMapLabels::Kind::City, 0, 1000) == 8.0f);
    REQUIRE(GeoMapLabels::CityTierFullVisibilityZoom(
                GeoMapLabels::Kind::Capital, 1, 1000) == 4.5f);

    REQUIRE(GeoMapLabels::CountryFullVisibilityZoom(0, 1.7f) == -1.0f);
    REQUIRE(GeoMapLabels::CountryFullVisibilityZoom(2, 1.7f) == -1.0f);
    REQUIRE(GeoMapLabels::CountryFullVisibilityZoom(2, 4.0f) == 1.3f);
    REQUIRE(GeoMapLabels::CountryFullVisibilityZoom(4, 1.7f) == 1.0f);
    REQUIRE(GeoMapLabels::CountryFullVisibilityZoom(4, 3.0f) == 1.0f);
    REQUIRE(GeoMapLabels::CountryFullVisibilityZoom(6, 1.7f) == 2.0f);
    REQUIRE(GeoMapLabels::CountryFullVisibilityZoom(9, 1.7f) == 2.5f);
    REQUIRE(GeoMapLabels::CountryFullVisibilityZoom(6, 5.7f) == Catch::Approx(3.0f));
    REQUIRE(GeoMapLabels::CountryFadeRange(2) == 0.75f);
    REQUIRE(GeoMapLabels::CountryFadeRange(9) == 1.0f);

    REQUIRE(GeoMapLabels::StateFullVisibilityZoom(2) == 2.75f);
    REQUIRE(GeoMapLabels::StateFullVisibilityZoom(4) == 3.45f);
    REQUIRE(GeoMapLabels::StateFullVisibilityZoom(10) == 5.55f);
    REQUIRE(GeoMapLabels::StateAreaLead(0.0f) == 0.0f);
    REQUIRE(GeoMapLabels::StateAreaLead(5.0f) == 0.0f);
    REQUIRE(GeoMapLabels::StateAreaLead(0.2f) == 0.0f);
    REQUIRE(GeoMapLabels::StateAreaLead(20.0f) == 0.8f);
    REQUIRE(GeoMapLabels::StateAreaLead(55.0f) == 1.0f);
    REQUIRE(GeoMapLabels::StateFullVisibilityZoom(0, 55.0f) == 1.75f);
    REQUIRE(GeoMapLabels::StateFullVisibilityZoom(3, 5.0f) == 3.10f);

    REQUIRE(GeoMapLabels::VisibilityFade(4.0f, 5.0f, 1.0f) == 0.0f);
    REQUIRE(GeoMapLabels::VisibilityFadeOut(
                GeoMapLabels::MajorGeographicHiddenZoom,
                GeoMapLabels::MajorGeographicHiddenZoom, 1.0f) == 0.0f);

    REQUIRE(GeoMapLabels::FixedPathLabelSpan(120.0f, 200.0f) == 120.0f);
    REQUIRE(GeoMapLabels::FixedPathLabelSpan(120.0f, 400.0f) == 120.0f);
    REQUIRE(GeoMapLabels::FixedPathLabelSpan(120.0f, 100.0f) == 120.0f);
    REQUIRE(GeoMapLabels::FixedPathNormalization(120.0f, 240.0f) == 0.5f);
    REQUIRE(GeoMapLabels::FixedPathNormalization(120.0f, 480.0f) == 0.25f);
    REQUIRE(240.0f * GeoMapLabels::FixedPathNormalization(120.0f, 240.0f) ==
            480.0f * GeoMapLabels::FixedPathNormalization(120.0f, 480.0f));
}
