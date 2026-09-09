#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include "jetstream/types.hh"

namespace Jetstream::Render::Components::GeoMapLabels {

inline constexpr F32 StateLineFadeStartZoom = -0.75f;
inline constexpr F32 StateLineFullZoom = 1.0f;
inline constexpr F32 StateLabelFadeStartZoom = 2.0f;
inline constexpr F32 StateLabelFullZoom = 2.75f;
inline constexpr F32 MajorGeographicHiddenZoom = 8.0f;

// The production renderer and policy tests use these same defaults and rules.
struct Tuning {
    F32 countryMajorFull = -1.0f;
    F32 countryMediumFull = 1.0f;
    F32 countrySmallFull = 2.0f;
    F32 countryLocalFull = 2.5f;
    F32 countrySourceLead = 2.7f;
    F32 stateMajorFull = StateLabelFullZoom;
    F32 stateRankStep = 0.35f;
    F32 stateAreaReference = 5.0f;
    F32 stateAreaLeadStep = 0.4f;
    F32 stateAreaMaxLead = 1.0f;
    F32 worldCityFull = 1.0f;
    F32 nationalCapitalFull = 2.0f;
    F32 megacityFull = 3.5f;
    F32 capitalFull = 4.5f;
    std::array<F32, 5> cityPopulationFull = {3.5f, 4.5f, 5.0f, 6.5f, 8.0f};
    F32 citySourceLead = 3.0f;
    std::array<F32, 3> airportFull = {5.0f, 7.0f, 9.0f};
    F32 pathLabelFull = 1.5f;
    F32 pathLabelDefaultHidden = MajorGeographicHiddenZoom;
};

enum class Kind : U8 {
    City = 0,
    Capital = 1,
    Country = 2,
    State = 3,
    Water = 4,
    Physical = 5,
    Airport = 6,
    River = 7,
    Lake = 8,
    Region = 9,
};

inline I32 ImportanceTier(Kind kind, U8 flags) {
    if (kind == Kind::Capital) {
        return flags & 8 ? 4 : 3;
    }
    if (kind == Kind::City) {
        return (flags & 2 ? 2 : 0) + (flags & 4 ? 1 : 0);
    }
    if (kind == Kind::Airport) {
        return flags & 1 ? 2 : flags & 2 ? 1 : 0;
    }
    return 0;
}

template <typename Label>
bool MoreImportant(const Label& a, const Label& b) {
    const I32 aTier = ImportanceTier(a.kind, a.flags);
    const I32 bTier = ImportanceTier(b.kind, b.flags);
    if (aTier != bTier) return aTier > bTier;
    if (a.scalerank != b.scalerank) return a.scalerank < b.scalerank;
    if (a.population != b.population) return a.population > b.population;
    if (a.minZoom != b.minZoom) return a.minZoom < b.minZoom;
    if (a.name != b.name) return a.name < b.name;
    if (a.lon != b.lon) return a.lon < b.lon;
    return a.lat < b.lat;
}

inline F32 FullVisibilityZoom(F32 sourceMinZoom,
                              I32 scalerank,
                              F32 layerFloor) {
    const F32 sourceZoom = sourceMinZoom > 0.0f
                               ? sourceMinZoom
                               : static_cast<F32>(scalerank) / 1.2f;
    return std::max(layerFloor, sourceZoom);
}

inline F32 CityTierFullVisibilityZoom(Kind kind,
                                      U8 flags,
                                      I32 population,
                                      const Tuning& tuning = {}) {
    if (kind != Kind::City && kind != Kind::Capital) return 0.0f;
    if (flags & 2) return tuning.worldCityFull;
    if (kind == Kind::Capital && flags & 8) return tuning.nationalCapitalFull;
    if (flags & 4) return tuning.megacityFull;
    if (kind == Kind::Capital) return tuning.capitalFull;
    if (population >= 5000000) return tuning.cityPopulationFull[0];
    if (population >= 1000000) return tuning.cityPopulationFull[1];
    if (population >= 250000) return tuning.cityPopulationFull[2];
    if (population >= 50000) return tuning.cityPopulationFull[3];
    return tuning.cityPopulationFull[4];
}

inline F32 CountryFullVisibilityZoom(I32 scalerank, F32 minZoom = 0.0f,
                                     const Tuning& tuning = {}) {
    // Country tiers follow the semantic zoom tree documented in GeoMap:
    // major countries first, then progressively smaller map units.
    // The Natural Earth min_zoom field adds fine-grained staggering within
    // each scalerank tier so that e.g. Belgium (min_zoom 4.0) does not
    // appear at the same zoom as the United States (min_zoom 1.7).
    F32 base;
    if (scalerank <= 2) base = tuning.countryMajorFull;
    else if (scalerank <= 4) base = tuning.countryMediumFull;
    else if (scalerank <= 6) base = tuning.countrySmallFull;
    else base = tuning.countryLocalFull;
    return std::max(base, minZoom - tuning.countrySourceLead);
}

inline F32 CountryFadeRange(I32 scalerank) {
    return scalerank <= 6 ? 0.75f : 1.0f;
}

// Zoom levels a state label is pulled forward by its size. `area` is the
// polygon area in latitude-corrected square degrees; units larger than the
// reference gain a lead per doubling, capped so labels never outrun borders
// by much. Texas-sized units earn the full lead, Bavaria-sized ones none.
inline F32 StateAreaLead(F32 area,
                         F32 referenceArea = 5.0f,
                         F32 leadStep = 0.4f,
                         F32 maxLead = 1.0f) {
    if (area <= 0.0f || referenceArea <= 0.0f) return 0.0f;
    const F32 doublings = std::log2(area / referenceArea);
    return std::clamp(doublings * leadStep, 0.0f, maxLead);
}

inline F32 StateFullVisibilityZoom(I32 scalerank, F32 area = 0.0f,
                                   const Tuning& tuning = {}) {
    // Natural Earth's state MIN_LABEL values target a flat map and extend as
    // high as zoom 18. Stage globe labels by rank instead: major first, then
    // progressively smaller administrative units through local-detail zooms.
    // Rank alone puts Texas and Rhode Island in the same tier, so polygon
    // area pulls country-sized units forward.
    const I32 rank = std::clamp(scalerank, 2, 10);
    return tuning.stateMajorFull + static_cast<F32>(rank - 2) * tuning.stateRankStep -
           StateAreaLead(area, tuning.stateAreaReference, tuning.stateAreaLeadStep,
                         tuning.stateAreaMaxLead);
}

template <typename Label>
F32 FullVisibilityZoom(const Label& label, const Tuning& tuning,
                       F32 layerFloor, F32 visibilityOffset) {
    if (label.kind == Kind::Country) {
        return CountryFullVisibilityZoom(label.scalerank, label.minZoom, tuning);
    }
    if (label.kind == Kind::State) {
        return StateFullVisibilityZoom(label.scalerank, label.area, tuning);
    }
    if (!label.path.empty()) return std::max(label.minZoom, tuning.pathLabelFull);
    const bool city = label.kind == Kind::City || label.kind == Kind::Capital;
    if (city && (label.flags & 2)) return tuning.worldCityFull;
    if (label.kind == Kind::Airport) {
        return label.flags & 1 ? tuning.airportFull[0]
            : label.flags & 2 ? tuning.airportFull[1] : tuning.airportFull[2];
    }
    F32 sourceZoom = FullVisibilityZoom(label.minZoom, label.scalerank, layerFloor);
    if (city) sourceZoom = std::max(layerFloor, sourceZoom - tuning.citySourceLead);
    const F32 tierZoom = CityTierFullVisibilityZoom(label.kind, label.flags,
                                                   label.population, tuning);
    return std::max(sourceZoom, tierZoom) + visibilityOffset;
}

inline F32 RiverFullVisibilityZoom(I32 scalerank) {
    return static_cast<F32>(std::max(0, scalerank - 3));
}

inline F32 RiverLineWidth(I32 scalerank) {
    const F32 rank = static_cast<F32>(std::clamp(scalerank, 0, 10));
    return 5.0f - rank * 0.35f;
}

inline F32 FixedPathLabelSpan(F32 layoutWidth, F32 pathLength) {
    return layoutWidth > 0.0f && pathLength > 0.0f ? layoutWidth : 0.0f;
}

inline F32 FixedPathNormalization(F32 labelSpan, F32 pathLength) {
    if (labelSpan <= 0.0f || pathLength <= 0.0f) return 0.0f;
    return labelSpan / pathLength;
}

inline F32 VisibilityFade(F32 zoom,
                          F32 fullVisibilityZoom,
                          F32 fadeRange = 1.0f) {
    fadeRange = std::max(fadeRange, 0.001f);
    const F32 startZoom = fullVisibilityZoom - fadeRange;
    constexpr F32 ZoomEpsilon = 1.0e-5f;
    if (zoom <= startZoom + ZoomEpsilon) return 0.0f;
    if (zoom >= fullVisibilityZoom - ZoomEpsilon) return 1.0f;
    const F32 t = (zoom - startZoom) / fadeRange;
    return t * t * (3.0f - 2.0f * t);
}

inline F32 VisibilityFadeOut(F32 zoom,
                             F32 maxVisibilityZoom,
                             F32 fadeRange = 1.0f) {
    if (maxVisibilityZoom <= 0.0f) return 1.0f;
    fadeRange = std::max(fadeRange, 0.001f);
    const F32 startZoom = maxVisibilityZoom - fadeRange;
    constexpr F32 ZoomEpsilon = 1.0e-5f;
    if (zoom <= startZoom + ZoomEpsilon) return 1.0f;
    if (zoom >= maxVisibilityZoom - ZoomEpsilon) return 0.0f;
    const F32 t = (zoom - startZoom) / fadeRange;
    const F32 smooth = t * t * (3.0f - 2.0f * t);
    return 1.0f - smooth;
}

// Shared by ordinary labels and path labels. Ensure a source's hidden zoom
// never cuts off the label before it can finish fading in.
inline F32 Visibility(F32 zoom, F32 fullZoom, F32 sourceMaxZoom, F32 fadeRange) {
    const F32 hiddenZoom = sourceMaxZoom > 0.0f
        ? std::max(sourceMaxZoom, fullZoom + fadeRange) : 0.0f;
    return std::min(VisibilityFade(zoom, fullZoom, fadeRange),
                    VisibilityFadeOut(zoom, hiddenZoom, fadeRange));
}

}  // namespace Jetstream::Render::Components::GeoMapLabels
