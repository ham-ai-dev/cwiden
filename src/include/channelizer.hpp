#pragma once
/**
 * channelizer.hpp — FFTW-based wideband channelizer.
 *
 * Takes raw IQ from an SDR, computes Welch PSD, detects peaks above the
 * noise floor, and extracts narrowband channels via NCO+FIR+decimate.
 */

#include <complex>
#include <vector>
#include <cstdint>
#include <fftw3.h>

struct DetectedChannel {
    double          center_freq_hz;  // absolute frequency
    float           snr_db;          // above noise floor
    float           peak_power_db;   // absolute PSD power
    std::vector<std::complex<float>> iq;  // narrowband baseband IQ
    float           bandwidth_hz;    // channel bandwidth (after decimation)
};

class Channelizer {
public:
    /**
     * @param sdr_sample_rate  Raw sample rate from SDR (e.g. 2000000)
     * @param sdr_center_freq  SDR center frequency in Hz
     * @param channel_bw       Per-channel bandwidth after decimation (default 500 Hz)
     * @param min_snr_db       Minimum SNR above noise floor to detect a channel
     * @param fft_size         FFT size for PSD computation
     */
    Channelizer(uint32_t sdr_sample_rate, double sdr_center_freq,
                float channel_bw = 500.0f, float min_snr_db = 6.0f,
                int fft_size = 4096);
    ~Channelizer();

    /**
     * Process a block of raw IQ samples.
     * Returns detected channels with extracted narrowband IQ.
     */
    std::vector<DetectedChannel> process(const std::complex<float>* samples, int count);

    /** Get the most recent PSD (for TUI spectrum display). */
    const std::vector<float>& last_psd_db() const { return psd_db_; }
    int psd_size() const { return fft_size_; }
    double freq_per_bin() const { return static_cast<double>(sdr_rate_) / fft_size_; }
    double center_freq() const { return center_freq_; }

    /** Update center frequency (e.g. after SDR retune). */
    void set_center_freq(double freq_hz) { center_freq_ = freq_hz; }

private:
    void compute_psd(const std::complex<float>* samples, int count);
    std::vector<int> detect_peaks();
    void extract_channel(const std::complex<float>* samples, int count,
                         double target_freq_hz, DetectedChannel& out);

    // FIR lowpass filter design
    void design_lpf(int num_taps, float cutoff);

    uint32_t sdr_rate_;
    double   center_freq_;
    float    channel_bw_;
    float    min_snr_db_;
    int      fft_size_;

    // FFTW
    fftwf_plan     fft_plan_ = nullptr;
    fftwf_complex* fft_in_   = nullptr;
    fftwf_complex* fft_out_  = nullptr;

    // PSD accumulation
    std::vector<float> psd_accum_;
    std::vector<float> psd_db_;
    std::vector<float> window_;   // Blackman-Harris
    int psd_seg_count_ = 0;

    // Channel extraction FIR
    std::vector<float> fir_taps_;
    int decimation_factor_ = 1;

    // Merge distance in bins
    int merge_bins_ = 0;
};
