#pragma once
/**
 * cw_classifier.hpp — 3-stage CW identification cascade.
 *
 * Stage 1: Spectral Shape — bandwidth, shape factor, two-tone detection
 * Stage 2: Amplitude Pattern — bimodality coefficient, on/off ratio
 * Stage 3: Temporal Rhythm — autocorrelation periodicity, WPM estimate
 *
 * Ported from deepspan's Python/CuPy classifiers to pure C++ with FFTW.
 */

#include <complex>
#include <vector>
#include <string>
#include <fftw3.h>

enum class SignalType {
    CW,
    SSB,
    RTTY,
    CARRIER,
    NOISE,
    UNKNOWN
};

struct ClassifyFeatures {
    // Stage 1
    float effective_bw = 0.0f;
    float shape_factor = 0.0f;
    float headroom_db = 0.0f;
    // Stage 2
    float bimodality_coeff = 0.0f;
    float on_off_ratio = 0.0f;
    // Stage 3
    float rhythm_score = 0.0f;
};

struct ClassifyResult {
    SignalType signal_class = SignalType::UNKNOWN;
    float      cw_confidence = 0.0f;  // 0.0–1.0
    int        stage_reached = 0;      // 1, 2, or 3
    float      wpm_estimate = 0.0f;    // only valid if is_cw && stage 3
    bool       is_cw = false;
    std::string verdict;               // human-readable reason
    ClassifyFeatures features;         // DSP measurements for logging
};

class CWClassifier {
public:
    CWClassifier();
    ~CWClassifier();

    /**
     * Run the full 3-stage cascade on narrowband baseband IQ.
     *
     * @param iq        Complex baseband samples (center ≈ 0 Hz)
     * @param count     Number of samples
     * @param sample_rate  Sample rate of the baseband (e.g. 500 Hz)
     * @return Classification result
     */
    ClassifyResult classify(const std::complex<float>* iq, int count, float sample_rate);

private:
    // Stage 1: Spectral Shape
    ClassifyResult stage1_spectral(const std::complex<float>* iq, int count, float fs);

    // Stage 2: Amplitude Pattern
    ClassifyResult stage2_amplitude(const std::complex<float>* iq, int count, float fs);

    // Stage 3: Temporal Rhythm (autocorrelation)
    ClassifyResult stage3_rhythm(const std::complex<float>* iq, int count, float fs);

    // Helpers
    float calc_bandwidth(const float* psd_db, const float* freqs, int nbins,
                         float peak_db, float threshold_db);

    // Reusable FFTW plans for stage 3
    // (created lazily based on input size)
    fftwf_plan  acorr_fwd_plan_ = nullptr;
    fftwf_plan  acorr_inv_plan_ = nullptr;
    int         acorr_plan_size_ = 0;
};

const char* signal_type_str(SignalType t);
