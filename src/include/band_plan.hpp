#pragma once
/**
 * band_plan.hpp — Ham radio band presets with CW sub-band ranges.
 *
 * Each band defines:
 *   - SDR center frequency (to place the CW sub-band in the passband)
 *   - CW sub-band lower/upper limits (for signal filtering)
 *   - Suggested sample rate
 */

#include <string>
#include <vector>
#include <cstdint>

struct BandPreset {
    std::string name;       // "160m", "80m", "40m", etc.
    double center_hz;       // SDR center frequency
    double cw_lo_hz;        // CW sub-band lower edge
    double cw_hi_hz;        // CW sub-band upper edge
    uint32_t sample_rate;   // Suggested sample rate

    bool contains(double freq_hz) const {
        return freq_hz >= cw_lo_hz && freq_hz <= cw_hi_hz;
    }
};

inline std::vector<BandPreset> get_band_presets() {
    return {
        // US amateur band plan — CW portions
        // Center freqs offset by +1.5 kHz to keep DC spike off popular CW spots
        {"160m", 1826500,  1800000,  1850000,  200000},
        {"80m",  3551500,  3500000,  3600000,  200000},
        {"40m",  7061500,  7000000,  7125000,  200000},
        {"30m",  10126500, 10100000, 10150000, 200000},
        {"20m",  14076500, 14000000, 14150000, 200000},
        {"17m",  18091500, 18068000, 18110000, 200000},
        {"15m",  21101500, 21000000, 21200000, 200000},
        {"12m",  24911500, 24890000, 24930000, 200000},
        {"10m",  28151500, 28000000, 28300000, 200000},
    };
}

inline const BandPreset* find_band(const std::string& name) {
    static auto presets = get_band_presets();
    for (auto& b : presets) {
        if (b.name == name) return &b;
    }
    return nullptr;
}

inline const BandPreset* find_band_for_freq(double freq_hz) {
    static auto presets = get_band_presets();
    for (auto& b : presets) {
        if (freq_hz >= b.cw_lo_hz - 50000 && freq_hz <= b.cw_hi_hz + 50000)
            return &b;
    }
    return nullptr;
}
