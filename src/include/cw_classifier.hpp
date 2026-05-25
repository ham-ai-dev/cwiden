#pragma once
/**
 * cw_classifier.hpp — CW identification with optional ML model.
 *
 * When no ML model is loaded, uses a 3-stage DSP cascade:
 *   Stage 1: Spectral Shape — bandwidth, shape factor, two-tone detection
 *   Stage 2: Amplitude Pattern — bimodality coefficient, on/off ratio
 *   Stage 3: Temporal Rhythm — autocorrelation periodicity, WPM estimate
 *
 * When an ML model is loaded (model.json), uses gradient-boosted tree
 * inference on the same feature vector for more accurate classification.
 */

#include <complex>
#include <vector>
#include <string>
#include <fftw3.h>

class MLClassifier;  // forward decl

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
    // Additional ML features
    float spectral_entropy = 0.0f;
    float peak_stability = 0.0f;
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
     * Attach an ML model for classification. When set, classify() uses
     * the ML model instead of hardcoded thresholds.
     */
    void set_ml_model(MLClassifier* model) { ml_model_ = model; }

    /**
     * Run classification on narrowband baseband IQ.
     * Uses ML model if available, otherwise DSP cascade.
     */
    ClassifyResult classify(const std::complex<float>* iq, int count, float sample_rate);

    /**
     * Extract all features without classifying (for ML training).
     * Always runs all 3 stages regardless of intermediate results.
     */
    ClassifyFeatures extract_features(const std::complex<float>* iq, int count, float sample_rate);

private:
    // Stage 1: Spectral Shape
    ClassifyResult stage1_spectral(const std::complex<float>* iq, int count, float fs);

    // Stage 2: Amplitude Pattern
    ClassifyResult stage2_amplitude(const std::complex<float>* iq, int count, float fs);

    // Stage 3: Temporal Rhythm (autocorrelation)
    ClassifyResult stage3_rhythm(const std::complex<float>* iq, int count, float fs);

    // Feature extractors
    float calc_spectral_entropy(const std::complex<float>* iq, int count, float fs);
    float calc_peak_stability(const std::complex<float>* iq, int count, float fs);

    // Helpers
    float calc_bandwidth(const float* psd_db, const float* freqs, int nbins,
                         float peak_db, float threshold_db);

    MLClassifier* ml_model_ = nullptr;

    // Reusable FFTW plans for stage 3
    fftwf_plan  acorr_fwd_plan_ = nullptr;
    fftwf_plan  acorr_inv_plan_ = nullptr;
    int         acorr_plan_size_ = 0;
};

const char* signal_type_str(SignalType t);
