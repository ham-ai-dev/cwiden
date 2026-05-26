#pragma once
/**
 * ml_classifier.hpp — ONNX-based CNN inference for CW classification.
 *
 * Loads a trained signal_classifier.onnx model (from deep-classifier)
 * and runs inference on spectrograms computed from raw IQ data.
 *
 * Pipeline: IQ → spectrogram (NFFT=64, HOP=16) → normalize → CNN → softmax
 */

#include <string>
#include <vector>
#include <complex>
#include <memory>

// Forward declare ORT types to avoid leaking headers
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
    struct MemoryInfo;
}

class MLClassifier {
public:
    MLClassifier();
    ~MLClassifier();

    // Non-copyable (ORT session is not copyable)
    MLClassifier(const MLClassifier&) = delete;
    MLClassifier& operator=(const MLClassifier&) = delete;

    /** Load ONNX model. Returns true on success. */
    bool load(const std::string& onnx_path);

    /** Is a model loaded and ready? */
    bool is_loaded() const { return loaded_; }

    /**
     * Classify narrowband IQ data.
     *
     * @param iq         Narrowband baseband IQ samples
     * @param count      Number of IQ samples
     * @param sample_rate Sample rate of the IQ data (Hz)
     * @return CW probability (0.0 = noise, 1.0 = CW)
     */
    float predict_iq(const std::complex<float>* iq, int count, float sample_rate);

    /**
     * Legacy interface: predict from pre-computed feature vector.
     * Falls back to 0.5 (no-op) since CNN doesn't use hand-crafted features.
     */
    float predict(const std::vector<float>& features) const;

    /** Model metadata */
    int n_classes() const { return n_classes_; }
    const std::vector<std::string>& class_names() const { return class_names_; }

private:
    /**
     * Compute spectrogram from IQ data.
     * Returns a flat vector in [freq_bins × time_frames] layout.
     */
    std::vector<float> compute_spectrogram(const std::complex<float>* iq, int count,
                                           float sample_rate);

    bool loaded_ = false;
    int n_classes_ = 2;
    std::vector<std::string> class_names_ = {"CW", "NOISE"};

    // Spectrogram parameters (must match training)
    int nfft_ = 64;
    int hop_ = 16;
    int target_time_frames_ = 256;

    // ONNX Runtime internals (opaque pointers to avoid header leakage)
    struct OrtState;
    std::unique_ptr<OrtState> ort_;
};
