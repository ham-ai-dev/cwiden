#include "ml_classifier.hpp"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fftw3.h>

// =========================================================================
// ONNX Runtime state (hidden from header)
// =========================================================================
struct MLClassifier::OrtState {
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info;

    OrtState()
        : env(ORT_LOGGING_LEVEL_WARNING, "cwiden_classifier")
        , mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {}
};

// =========================================================================
// Construction / destruction
// =========================================================================
MLClassifier::MLClassifier() {}

MLClassifier::~MLClassifier() = default;

// =========================================================================
// Load ONNX model
// =========================================================================
bool MLClassifier::load(const std::string& onnx_path) {
    try {
        ort_ = std::make_unique<OrtState>();

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        ort_->session = std::make_unique<Ort::Session>(
            ort_->env, onnx_path.c_str(), opts
        );

        // Read input shape to verify
        auto input_info = ort_->session->GetInputTypeInfo(0);
        auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        // Expected: [batch, 1, 64, 256]
        if (shape.size() == 4) {
            nfft_ = static_cast<int>(shape[2]);
            target_time_frames_ = static_cast<int>(shape[3]);
        }

        // Read output shape for n_classes
        auto output_info = ort_->session->GetOutputTypeInfo(0);
        auto out_tensor = output_info.GetTensorTypeAndShapeInfo();
        auto out_shape = out_tensor.GetShape();
        if (out_shape.size() == 2) {
            n_classes_ = static_cast<int>(out_shape[1]);
        }

        loaded_ = true;
        std::cerr << "MLClassifier: loaded ONNX model from " << onnx_path
                  << " (input: [B,1," << nfft_ << "," << target_time_frames_
                  << "], classes: " << n_classes_ << ")" << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "MLClassifier ONNX error: " << e.what() << std::endl;
        ort_.reset();
        loaded_ = false;
        return false;
    }
}

// =========================================================================
// Compute spectrogram from IQ (matches Python training pipeline exactly)
// =========================================================================
std::vector<float> MLClassifier::compute_spectrogram(
    const std::complex<float>* iq, int count, float sample_rate)
{
    int nperseg = nfft_;
    int noverlap = nperseg - hop_;
    int step = hop_;

    // Number of time frames
    int n_frames = 0;
    if (count >= nperseg) {
        n_frames = (count - nperseg) / step + 1;
    }
    if (n_frames < 1) n_frames = 1;

    // Allocate spectrogram: [nfft_ × n_frames]
    std::vector<float> spec(nfft_ * n_frames, 0.0f);

    // FFTW plan
    fftwf_complex* fin  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nperseg);
    fftwf_complex* fout = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nperseg);
    fftwf_plan plan = fftwf_plan_dft_1d(nperseg, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int frame = 0; frame < n_frames; frame++) {
        int offset = frame * step;

        // Fill FFT input (no window — matches scipy default for mode="magnitude")
        for (int i = 0; i < nperseg; i++) {
            if (offset + i < count) {
                fin[i][0] = iq[offset + i].real();
                fin[i][1] = iq[offset + i].imag();
            } else {
                fin[i][0] = 0.0f;
                fin[i][1] = 0.0f;
            }
        }

        fftwf_execute(plan);

        // Magnitude spectrum, FFT-shifted (zero freq → center)
        int half = nperseg / 2;
        for (int i = 0; i < nperseg; i++) {
            int src = (i + half) % nperseg;
            float mag = std::sqrt(fout[src][0] * fout[src][0] +
                                  fout[src][1] * fout[src][1]);
            spec[i * n_frames + frame] = mag;  // [freq, time] layout
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(fin);
    fftwf_free(fout);

    // Log scale
    for (auto& v : spec) {
        v = 20.0f * std::log10(v + 1e-12f);
    }

    // Normalize: zero mean, unit variance (per-clip, matching training)
    float sum = 0.0f;
    for (auto v : spec) sum += v;
    float mean = sum / spec.size();

    float var_sum = 0.0f;
    for (auto v : spec) var_sum += (v - mean) * (v - mean);
    float std_dev = std::sqrt(var_sum / spec.size()) + 1e-8f;

    for (auto& v : spec) {
        v = (v - mean) / std_dev;
    }

    // Pad/truncate time dimension to target_time_frames_
    // Current layout: [nfft_ × n_frames], need [nfft_ × target_time_frames_]
    std::vector<float> output(nfft_ * target_time_frames_, 0.0f);

    if (n_frames >= target_time_frames_) {
        // Truncate (take center)
        int start = (n_frames - target_time_frames_) / 2;
        for (int f = 0; f < nfft_; f++) {
            for (int t = 0; t < target_time_frames_; t++) {
                output[f * target_time_frames_ + t] = spec[f * n_frames + start + t];
            }
        }
    } else {
        // Pad with zeros
        for (int f = 0; f < nfft_; f++) {
            for (int t = 0; t < n_frames; t++) {
                output[f * target_time_frames_ + t] = spec[f * n_frames + t];
            }
        }
    }

    return output;
}

// =========================================================================
// Predict from raw IQ
// =========================================================================
float MLClassifier::predict_iq(const std::complex<float>* iq, int count,
                                float sample_rate) {
    if (!loaded_ || !ort_ || !ort_->session) return 0.5f;

    // The CNN was trained on 1000 Hz sample rate IQ.
    // cwiden's channelizer may output at a different rate (e.g. 200 Hz).
    // Resample to training rate if needed.
    const float training_sr = 1000.0f;
    const std::complex<float>* iq_for_spec = iq;
    int count_for_spec = count;
    std::vector<std::complex<float>> resampled;

    if (std::abs(sample_rate - training_sr) > 1.0f && sample_rate > 0.0f) {
        float ratio = training_sr / sample_rate;
        int new_count = static_cast<int>(count * ratio);
        resampled.resize(new_count);

        // Linear interpolation resampling
        for (int i = 0; i < new_count; i++) {
            float src_pos = i / ratio;
            int idx = static_cast<int>(src_pos);
            float frac = src_pos - idx;
            if (idx + 1 < count) {
                resampled[i] = iq[idx] * (1.0f - frac) + iq[idx + 1] * frac;
            } else if (idx < count) {
                resampled[i] = iq[idx];
            }
        }
        iq_for_spec = resampled.data();
        count_for_spec = new_count;
    }

    // Compute spectrogram at training sample rate
    auto spec = compute_spectrogram(iq_for_spec, count_for_spec, training_sr);

    // Shape: [1, 1, nfft_, target_time_frames_]
    std::array<int64_t, 4> shape = {1, 1, nfft_, target_time_frames_};

    auto input_tensor = Ort::Value::CreateTensor<float>(
        ort_->mem_info, spec.data(), spec.size(),
        shape.data(), shape.size()
    );

    // Run inference
    const char* input_names[] = {"spectrogram"};
    const char* output_names[] = {"logits"};

    try {
        auto results = ort_->session->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );

        float* logits = results[0].GetTensorMutableData<float>();

        // Softmax to get probabilities
        float max_logit = *std::max_element(logits, logits + n_classes_);
        float exp_sum = 0.0f;
        std::vector<float> probs(n_classes_);
        for (int i = 0; i < n_classes_; i++) {
            probs[i] = std::exp(logits[i] - max_logit);
            exp_sum += probs[i];
        }
        for (auto& p : probs) p /= exp_sum;

        // Return CW probability (index 0)
        return probs[0];

    } catch (const Ort::Exception& e) {
        std::cerr << "MLClassifier inference error: " << e.what() << std::endl;
        return 0.5f;
    }
}

// =========================================================================
// Legacy predict (feature-vector interface, kept for compatibility)
// =========================================================================
float MLClassifier::predict(const std::vector<float>& features) const {
    // CNN doesn't use hand-crafted features — return neutral
    (void)features;
    return 0.5f;
}
