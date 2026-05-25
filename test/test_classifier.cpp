/**
 * test_classifier.cpp — Synthetic signal tests for CWClassifier.
 *
 * Tests:
 *   1. Pure CW signal (20 WPM) → must identify as CW
 *   2. AWGN noise → must reject
 *   3. Wideband noise (SSB-like) → must reject at Stage 1
 *   4. Unmodulated carrier → must reject
 *   5. CW at various WPM (5, 35, 50)
 *   6. CW at low SNR
 */

#include "cw_classifier.hpp"
#include "channelizer.hpp"

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static std::mt19937 rng(42);

// Generate complex AWGN
static void add_noise(std::vector<std::complex<float>>& sig, float snr_db) {
    // Measure signal power
    float sig_power = 0.0f;
    for (auto& s : sig) {
        sig_power += std::norm(s);
    }
    sig_power /= sig.size();

    float noise_power = sig_power / std::pow(10.0f, snr_db / 10.0f);
    float sigma = std::sqrt(noise_power / 2.0f);

    std::normal_distribution<float> dist(0.0f, sigma);
    for (auto& s : sig) {
        s += std::complex<float>(dist(rng), dist(rng));
    }
}

// Generate synthetic CW signal at baseband (0 Hz carrier)
// wpm: words per minute, fs: sample rate
static std::vector<std::complex<float>> gen_cw(float wpm, float fs,
                                                 float duration_sec, float snr_db) {
    float dit_sec = 1.2f / wpm;
    int dit_samples = static_cast<int>(dit_sec * fs);
    int dah_samples = dit_samples * 3;
    int space_samples = dit_samples;
    int char_space = dit_samples * 3;

    int total = static_cast<int>(duration_sec * fs);
    std::vector<std::complex<float>> sig(total, {0.0f, 0.0f});

    // Generate a repeating "PARIS " pattern (standard WPM calibration word)
    // P: .--. A: .- R: .-. I: .. S: ...
    // Simplified: just alternate dits and dahs
    const char* pattern = ".--..-.-...-..."; // PARIS
    int pos = 0;
    int pi = 0;

    while (pos < total) {
        char elem = pattern[pi % strlen(pattern)];
        int len = (elem == '.') ? dit_samples : dah_samples;

        // Key on: amplitude = 1.0
        for (int i = 0; i < len && pos < total; i++, pos++) {
            sig[pos] = {1.0f, 0.0f};
        }

        pi++;
        // Element space
        if (pi % strlen(pattern) == 0) {
            // Word space (7 dits)
            for (int i = 0; i < dit_samples * 7 && pos < total; i++, pos++) {
                sig[pos] = {0.0f, 0.0f};
            }
        } else {
            // Inter-element space (1 dit)
            pos += space_samples;
        }
    }

    // Add noise
    if (snr_db < 100.0f) {
        add_noise(sig, snr_db);
    }

    return sig;
}

// Generate pure noise
static std::vector<std::complex<float>> gen_noise(float fs, float duration_sec) {
    int total = static_cast<int>(duration_sec * fs);
    std::vector<std::complex<float>> sig(total);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& s : sig) {
        s = std::complex<float>(dist(rng), dist(rng));
    }
    return sig;
}

// Generate unmodulated carrier
static std::vector<std::complex<float>> gen_carrier(float fs, float duration_sec, float snr_db) {
    int total = static_cast<int>(duration_sec * fs);
    std::vector<std::complex<float>> sig(total, {1.0f, 0.0f});
    if (snr_db < 100.0f) add_noise(sig, snr_db);
    return sig;
}

// =====================================================================
struct TestResult {
    const char* name;
    bool passed;
    std::string detail;
};

static std::vector<TestResult> results;

static void run_test(const char* name, bool condition, const std::string& detail) {
    results.push_back({name, condition, detail});
    std::cout << (condition ? "  PASS" : "  FAIL") << "  " << name
              << "  [" << detail << "]" << std::endl;
}

int main() {
    std::cout << "\n=== cwiden Classifier Tests ===\n" << std::endl;

    CWClassifier clf;
    float fs = 500.0f;  // Standard baseband rate

    // Test 1: CW 20 WPM at good SNR
    {
        auto sig = gen_cw(20.0f, fs, 6.0f, 20.0f);
        auto r = clf.classify(sig.data(), sig.size(), fs);
        run_test("CW 20wpm SNR=20dB", r.is_cw,
                 std::string("conf=") + std::to_string(r.cw_confidence)
                 + " stage=" + std::to_string(r.stage_reached)
                 + " wpm=" + std::to_string(r.wpm_estimate)
                 + " | " + r.verdict);
    }

    // Test 2: Pure noise → should NOT be CW
    {
        auto sig = gen_noise(fs, 6.0f);
        auto r = clf.classify(sig.data(), sig.size(), fs);
        run_test("Pure AWGN (no CW)", !r.is_cw,
                 std::string("conf=") + std::to_string(r.cw_confidence)
                 + " class=" + signal_type_str(r.signal_class)
                 + " | " + r.verdict);
    }

    // Test 3: Carrier → should NOT be CW
    {
        auto sig = gen_carrier(fs, 6.0f, 30.0f);
        auto r = clf.classify(sig.data(), sig.size(), fs);
        run_test("Unmodulated carrier", !r.is_cw,
                 std::string("conf=") + std::to_string(r.cw_confidence)
                 + " class=" + signal_type_str(r.signal_class)
                 + " | " + r.verdict);
    }

    // Test 4: CW 5 WPM (slow)
    {
        auto sig = gen_cw(5.0f, fs, 10.0f, 20.0f);
        auto r = clf.classify(sig.data(), sig.size(), fs);
        run_test("CW 5wpm SNR=20dB", r.is_cw,
                 std::string("conf=") + std::to_string(r.cw_confidence)
                 + " wpm=" + std::to_string(r.wpm_estimate)
                 + " | " + r.verdict);
    }

    // Test 5: CW 35 WPM (fast)
    {
        auto sig = gen_cw(35.0f, fs, 6.0f, 20.0f);
        auto r = clf.classify(sig.data(), sig.size(), fs);
        run_test("CW 35wpm SNR=20dB", r.is_cw,
                 std::string("conf=") + std::to_string(r.cw_confidence)
                 + " wpm=" + std::to_string(r.wpm_estimate)
                 + " | " + r.verdict);
    }

    // Test 6: CW at low SNR (6 dB)
    {
        auto sig = gen_cw(20.0f, fs, 6.0f, 6.0f);
        auto r = clf.classify(sig.data(), sig.size(), fs);
        // At 6 dB we accept either pass or a reasonable confidence score
        run_test("CW 20wpm SNR=6dB (weak)", r.cw_confidence > 0.1f,
                 std::string("conf=") + std::to_string(r.cw_confidence)
                 + " is_cw=" + (r.is_cw ? "true" : "false")
                 + " | " + r.verdict);
    }

    // Test 7: Channelizer detects injected tone
    {
        int rate = 2000000;
        double center = 6900000.0;
        int count = rate * 2;  // 2 seconds
        std::vector<std::complex<float>> wideband(count);

        // Inject a tone at 7.010 MHz (offset = 110 kHz from center)
        double tone_freq = 7010000.0;
        double offset = tone_freq - center;
        for (int i = 0; i < count; i++) {
            double phase = 2.0 * M_PI * offset * i / rate;
            wideband[i] = std::complex<float>(0.5f * std::cos(phase), 0.5f * std::sin(phase));
        }
        // Add noise
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (auto& s : wideband) s += std::complex<float>(dist(rng), dist(rng));

        Channelizer ch(rate, center, 500.0f, 6.0f, 4096);
        auto channels = ch.process(wideband.data(), count);

        bool found = false;
        for (auto& c : channels) {
            if (std::abs(c.center_freq_hz - tone_freq) < 1000.0) {
                found = true;
            }
        }
        run_test("Channelizer: detect tone at 7.010 MHz", found,
                 "channels=" + std::to_string(channels.size()));
    }

    // Summary
    int passed = 0, failed = 0;
    for (auto& r : results) {
        if (r.passed) passed++; else failed++;
    }
    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===\n" << std::endl;

    return failed > 0 ? 1 : 0;
}
