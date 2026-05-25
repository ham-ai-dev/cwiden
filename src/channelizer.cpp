#include "channelizer.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Channelizer::Channelizer(uint32_t sdr_sample_rate, double sdr_center_freq,
                         float channel_bw, float min_snr_db, int fft_size)
    : sdr_rate_(sdr_sample_rate),
      center_freq_(sdr_center_freq),
      channel_bw_(channel_bw),
      min_snr_db_(min_snr_db),
      fft_size_(fft_size)
{
    // Allocate FFTW buffers
    fft_in_  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size_);
    fft_out_ = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size_);
    fft_plan_ = fftwf_plan_dft_1d(fft_size_, fft_in_, fft_out_,
                                   FFTW_FORWARD, FFTW_MEASURE);

    psd_accum_.resize(fft_size_, 0.0f);
    psd_db_.resize(fft_size_, -120.0f);

    // Blackman-Harris window
    window_.resize(fft_size_);
    for (int i = 0; i < fft_size_; i++) {
        double x = 2.0 * M_PI * i / (fft_size_ - 1);
        window_[i] = static_cast<float>(
            0.35875 - 0.48829 * cos(x) + 0.14128 * cos(2.0 * x) - 0.01168 * cos(3.0 * x));
    }

    // Decimation factor: sdr_rate → channel_bw
    decimation_factor_ = std::max(1, static_cast<int>(sdr_rate_ / channel_bw_));

    // FIR lowpass for channel extraction
    int num_taps = std::min(201, decimation_factor_ * 4 + 1);
    if (num_taps % 2 == 0) num_taps++;
    design_lpf(num_taps, 1.0f / decimation_factor_);

    // Merge distance: 100 Hz in bins
    merge_bins_ = std::max(1, static_cast<int>(100.0 / (static_cast<double>(sdr_rate_) / fft_size_)));
}

Channelizer::~Channelizer() {
    if (fft_plan_) fftwf_destroy_plan(fft_plan_);
    if (fft_in_)   fftwf_free(fft_in_);
    if (fft_out_)  fftwf_free(fft_out_);
}

void Channelizer::design_lpf(int num_taps, float cutoff) {
    fir_taps_.resize(num_taps);
    int mid = num_taps / 2;
    float sum = 0.0f;
    for (int i = 0; i < num_taps; i++) {
        float x = static_cast<float>(i - mid);
        float sinc = (std::fabs(x) < 1e-10f)
            ? 2.0f * cutoff
            : std::sin(2.0f * static_cast<float>(M_PI) * cutoff * x)
              / (static_cast<float>(M_PI) * x);
        // Blackman window
        float win = 0.42f - 0.50f * std::cos(2.0f * static_cast<float>(M_PI) * i / num_taps)
                          + 0.08f * std::cos(4.0f * static_cast<float>(M_PI) * i / num_taps);
        fir_taps_[i] = sinc * win;
        sum += fir_taps_[i];
    }
    for (auto& t : fir_taps_) t /= sum;
}

void Channelizer::compute_psd(const std::complex<float>* samples, int count) {
    // Welch method: overlapping segments, windowed, averaged
    int overlap = fft_size_ / 2;
    int step = fft_size_ - overlap;
    int n_segments = 0;

    std::fill(psd_accum_.begin(), psd_accum_.end(), 0.0f);

    for (int offset = 0; offset + fft_size_ <= count; offset += step) {
        // Window + load
        for (int i = 0; i < fft_size_; i++) {
            fft_in_[i][0] = samples[offset + i].real() * window_[i];
            fft_in_[i][1] = samples[offset + i].imag() * window_[i];
        }

        fftwf_execute(fft_plan_);

        // Accumulate |X[k]|²
        for (int i = 0; i < fft_size_; i++) {
            float re = fft_out_[i][0];
            float im = fft_out_[i][1];
            psd_accum_[i] += re * re + im * im;
        }
        n_segments++;
    }

    if (n_segments == 0) return;

    // Average and convert to dB, with FFT shift (DC in center)
    float norm = 1.0f / n_segments;
    int half = fft_size_ / 2;
    for (int i = 0; i < fft_size_; i++) {
        int src = (i + half) % fft_size_;  // FFT shift
        float power = psd_accum_[src] * norm;
        psd_db_[i] = 10.0f * std::log10(power + 1e-12f);
    }
    psd_seg_count_ = n_segments;
}

std::vector<int> Channelizer::detect_peaks() {
    // Noise floor: median of PSD
    std::vector<float> sorted_psd = psd_db_;
    std::sort(sorted_psd.begin(), sorted_psd.end());
    float noise_floor = sorted_psd[sorted_psd.size() / 2];

    float threshold = noise_floor + min_snr_db_;

    // Find bins above threshold
    std::vector<int> peak_bins;
    int half = fft_size_ / 2;

    // Skip DC region (±5 bins around center)
    int dc_exclude = 5;

    for (int i = 0; i < fft_size_; i++) {
        // Skip DC spike region
        if (std::abs(i - half) < dc_exclude) continue;

        if (psd_db_[i] > threshold) {
            // Check if local maximum (higher than neighbors)
            bool is_peak = true;
            for (int j = std::max(0, i - 1); j <= std::min(fft_size_ - 1, i + 1); j++) {
                if (j != i && psd_db_[j] > psd_db_[i]) {
                    is_peak = false;
                    break;
                }
            }
            if (is_peak) {
                peak_bins.push_back(i);
            }
        }
    }

    // Merge peaks within merge_bins_ distance (keep strongest)
    std::vector<int> merged;
    for (int i = 0; i < static_cast<int>(peak_bins.size()); i++) {
        if (merged.empty() || peak_bins[i] - merged.back() > merge_bins_) {
            merged.push_back(peak_bins[i]);
        } else {
            // Keep the stronger one
            if (psd_db_[peak_bins[i]] > psd_db_[merged.back()]) {
                merged.back() = peak_bins[i];
            }
        }
    }

    return merged;
}

void Channelizer::extract_channel(const std::complex<float>* samples, int count,
                                   double target_freq_hz, DetectedChannel& out) {
    double freq_offset = target_freq_hz - center_freq_;

    // NCO frequency shift to baseband
    std::vector<std::complex<float>> shifted(count);
    double phase = 0.0;
    double phase_inc = 2.0 * M_PI * freq_offset / sdr_rate_;

    for (int i = 0; i < count; i++) {
        float cos_v = std::cos(phase);
        float sin_v = std::sin(phase);
        std::complex<float> nco(cos_v, -sin_v);
        shifted[i] = samples[i] * nco;
        phase += phase_inc;
        if (phase > 2.0 * M_PI) phase -= 2.0 * M_PI;
        if (phase < -2.0 * M_PI) phase += 2.0 * M_PI;
    }

    // FIR filter + decimate
    int output_len = count / decimation_factor_;
    out.iq.resize(output_len);
    int taps = static_cast<int>(fir_taps_.size());

    for (int n = 0; n < output_len; n++) {
        int center = n * decimation_factor_;
        std::complex<float> sum(0.0f, 0.0f);
        for (int k = 0; k < taps; k++) {
            int idx = center - taps / 2 + k;
            if (idx >= 0 && idx < count) {
                sum += fir_taps_[k] * shifted[idx];
            }
        }
        out.iq[n] = sum;
    }

    out.center_freq_hz = target_freq_hz;
    out.bandwidth_hz = static_cast<float>(sdr_rate_) / decimation_factor_;
}

std::vector<DetectedChannel> Channelizer::process(const std::complex<float>* samples, int count) {
    if (count < fft_size_) return {};

    compute_psd(samples, count);

    auto peaks = detect_peaks();

    // Noise floor for SNR calculation
    std::vector<float> sorted_psd = psd_db_;
    std::sort(sorted_psd.begin(), sorted_psd.end());
    float noise_floor = sorted_psd[sorted_psd.size() / 2];

    std::vector<DetectedChannel> channels;
    int half = fft_size_ / 2;
    double hz_per_bin = static_cast<double>(sdr_rate_) / fft_size_;

    for (int bin : peaks) {
        // Convert bin to absolute frequency
        // After FFT shift: bin 0 = center_freq - sdr_rate/2
        //                  bin N/2 = center_freq
        //                  bin N-1 = center_freq + sdr_rate/2 - hz_per_bin
        double freq_hz = center_freq_ + (bin - half) * hz_per_bin;
        float snr = psd_db_[bin] - noise_floor;

        DetectedChannel ch;
        ch.snr_db = snr;
        ch.peak_power_db = psd_db_[bin];
        extract_channel(samples, count, freq_hz, ch);

        channels.push_back(std::move(ch));
    }

    return channels;
}
