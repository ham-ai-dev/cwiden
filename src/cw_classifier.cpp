#include "cw_classifier.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <sstream>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const char* signal_type_str(SignalType t) {
    switch (t) {
        case SignalType::CW:      return "CW";
        case SignalType::SSB:     return "SSB";
        case SignalType::RTTY:    return "RTTY";
        case SignalType::CARRIER: return "CARRIER";
        case SignalType::NOISE:   return "NOISE";
        case SignalType::UNKNOWN: return "UNKNOWN";
    }
    return "?";
}

CWClassifier::CWClassifier() {}

CWClassifier::~CWClassifier() {
    if (acorr_fwd_plan_) fftwf_destroy_plan(acorr_fwd_plan_);
    if (acorr_inv_plan_) fftwf_destroy_plan(acorr_inv_plan_);
}

// =========================================================================
// Public entry point
// =========================================================================
ClassifyResult CWClassifier::classify(const std::complex<float>* iq, int count,
                                       float sample_rate) {
    if (count < 32) {
        return {SignalType::UNKNOWN, 0.0f, 0, 0.0f, false, "Too few samples"};
    }

    ClassifyFeatures features;

    // Stage 1
    auto r1 = stage1_spectral(iq, count, sample_rate);
    features.effective_bw = r1.features.effective_bw;
    features.shape_factor = r1.features.shape_factor;
    features.headroom_db = r1.features.headroom_db;
    if (!r1.is_cw) { r1.features = features; return r1; }

    // Stage 2
    auto r2 = stage2_amplitude(iq, count, sample_rate);
    features.bimodality_coeff = r2.features.bimodality_coeff;
    features.on_off_ratio = r2.features.on_off_ratio;
    if (!r2.is_cw) {
        r2.cw_confidence = r1.cw_confidence * 0.3f;
        r2.features = features;
        return r2;
    }

    // Stage 3
    auto r3 = stage3_rhythm(iq, count, sample_rate);
    features.rhythm_score = r3.features.rhythm_score;
    r3.features = features;

    // Combine confidences from all stages
    if (r3.is_cw) {
        r3.cw_confidence = 0.2f * r1.cw_confidence + 0.3f * r2.cw_confidence
                         + 0.5f * r3.cw_confidence;
    } else {
        r3.cw_confidence = 0.2f * r1.cw_confidence + 0.3f * r2.cw_confidence
                         + 0.5f * r3.cw_confidence * 0.3f;
    }
    return r3;
}

// =========================================================================
// Stage 1: Spectral Shape
// =========================================================================
ClassifyResult CWClassifier::stage1_spectral(const std::complex<float>* iq,
                                              int count, float fs) {
    // Welch PSD via FFTW
    int nperseg = std::min(1024, count);
    if (nperseg < 16) {
        return {SignalType::UNKNOWN, 0.0f, 1, 0.0f, false, "S1: chunk too short"};
    }

    // Round to power of 2 for FFT efficiency
    int fft_n = 1;
    while (fft_n < nperseg) fft_n <<= 1;
    if (fft_n > count) fft_n >>= 1;
    nperseg = fft_n;

    // Blackman-Harris window
    std::vector<float> win(nperseg);
    for (int i = 0; i < nperseg; i++) {
        double x = 2.0 * M_PI * i / (nperseg - 1);
        win[i] = static_cast<float>(0.35875 - 0.48829*cos(x)
                 + 0.14128*cos(2.0*x) - 0.01168*cos(3.0*x));
    }

    // Compute PSD (single segment for simplicity at 500 Hz rate)
    fftwf_complex* fin  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nperseg);
    fftwf_complex* fout = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nperseg);
    fftwf_plan plan = fftwf_plan_dft_1d(nperseg, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE);

    // Average multiple segments
    std::vector<float> psd(nperseg, 0.0f);
    int overlap = nperseg / 2;
    int step = nperseg - overlap;
    int n_seg = 0;

    for (int off = 0; off + nperseg <= count; off += step) {
        for (int i = 0; i < nperseg; i++) {
            fin[i][0] = iq[off + i].real() * win[i];
            fin[i][1] = iq[off + i].imag() * win[i];
        }
        fftwf_execute(plan);
        for (int i = 0; i < nperseg; i++) {
            psd[i] += fout[i][0]*fout[i][0] + fout[i][1]*fout[i][1];
        }
        n_seg++;
    }

    fftwf_destroy_plan(plan);
    fftwf_free(fin);
    fftwf_free(fout);

    if (n_seg == 0) {
        return {SignalType::UNKNOWN, 0.0f, 1, 0.0f, false, "S1: no segments"};
    }

    // Normalize, convert to dB, FFT-shift
    float norm = 1.0f / n_seg;
    std::vector<float> psd_db(nperseg);
    std::vector<float> freqs(nperseg);
    int half = nperseg / 2;
    float df = fs / nperseg;

    for (int i = 0; i < nperseg; i++) {
        int src = (i + half) % nperseg;
        psd_db[i] = 10.0f * std::log10(psd[src] * norm + 1e-12f);
        freqs[i] = (i - half) * df;
    }

    float max_psd = *std::max_element(psd_db.begin(), psd_db.end());

    // Bandwidth at various levels
    float bw_3  = calc_bandwidth(psd_db.data(), freqs.data(), nperseg, max_psd, max_psd - 3.0f);
    float bw_20 = calc_bandwidth(psd_db.data(), freqs.data(), nperseg, max_psd, max_psd - 20.0f);

    // Guard against bw_3 ≈ 0
    float min_bw = std::max(1e-3f, df);
    if (bw_3 < min_bw) bw_3 = min_bw;
    float shape_factor = bw_20 / bw_3;

    // Noise floor
    std::vector<float> sorted_psd = psd_db;
    std::sort(sorted_psd.begin(), sorted_psd.end());
    float noise_floor = sorted_psd[sorted_psd.size() / 2];
    float headroom = max_psd - noise_floor;

    // Adaptive effective BW for weak signals
    float effective_bw;
    if (headroom < 25.0f) {
        float level = max_psd - std::max(3.0f, headroom / 2.0f);
        effective_bw = calc_bandwidth(psd_db.data(), freqs.data(), nperseg, max_psd, level);
    } else {
        effective_bw = bw_20;
    }

    // --- Two-tone RTTY detection ---
    float threshold_6 = max_psd - 6.0f;
    int dist_bins = std::max(1, static_cast<int>(100.0f / df));
    // Simple peak search for two-tone
    bool has_two_tone = false;
    std::vector<int> tone_peaks;
    for (int i = 1; i < nperseg - 1; i++) {
        if (psd_db[i] > threshold_6 && psd_db[i] >= psd_db[i-1] && psd_db[i] >= psd_db[i+1]) {
            if (tone_peaks.empty() || i - tone_peaks.back() >= dist_bins) {
                tone_peaks.push_back(i);
            }
        }
    }
    if (tone_peaks.size() == 2) {
        float spacing = std::abs(freqs[tone_peaks[0]] - freqs[tone_peaks[1]]);
        bool is_170 = spacing >= 150.0f && spacing <= 200.0f;
        bool is_850 = spacing >= 800.0f && spacing <= 900.0f;
        has_two_tone = is_170 || is_850;
    }

    // --- Fast exits ---
    std::ostringstream verdict;

    // SSB: signal occupies > 50% of channel
    if (effective_bw > fs * 0.5f) {
        verdict << "S1: SSB (eff_bw=" << effective_bw << "Hz > " << fs*0.5f << "Hz)";
        return {SignalType::SSB, 0.0f, 1, 0.0f, false, verdict.str()};
    }

    // Carrier: very narrow — but only if amplitude is NOT bimodal
    // Fast CW at baseband (0 Hz carrier) can have BW < 5 Hz spectrally
    // but will still show on/off keying in the amplitude domain.
    // So we do a quick bimodality check before declaring CARRIER.
    if (effective_bw < 5.0f) {
        // Quick amplitude check
        float env_mu = 0.0f;
        for (int i = 0; i < count; i++) env_mu += std::abs(iq[i]);
        env_mu /= count;
        float env_sigma = 0.0f;
        for (int i = 0; i < count; i++) {
            float d = std::abs(iq[i]) - env_mu;
            env_sigma += d * d;
        }
        env_sigma = std::sqrt(env_sigma / count);
        float cv = (env_mu > 1e-6f) ? env_sigma / env_mu : 0.0f;
        // CV < 0.1 means truly flat → carrier.  CW keying has CV > 0.3.
        if (cv < 0.15f) {
            verdict << "S1: CARRIER (eff_bw=" << effective_bw << "Hz, CV=" << cv << ")";
            return {SignalType::CARRIER, 0.0f, 1, 0.0f, false, verdict.str()};
        }
        // Otherwise: narrow but keyed — let it pass to Stage 2/3
    }

    // RTTY
    if (has_two_tone) {
        return {SignalType::RTTY, 0.0f, 1, 0.0f, false, "S1: RTTY two-tone detected"};
    }

    // CW bandwidth check: relax lower bound to 2 Hz for fast CW
    float cw_bw_limit = std::max(800.0f, fs * 0.4f);
    bool pass_bw = effective_bw >= 2.0f && effective_bw <= cw_bw_limit;

    // Shape factor check (adaptive to headroom)
    float hf = std::clamp((headroom - 3.0f) / 27.0f, 0.0f, 1.0f);
    float shape_limit = 5000.0f - hf * 4950.0f;
    bool pass_shape = shape_factor >= 1.0f && shape_factor <= shape_limit;

    bool passed = pass_bw && pass_shape;

    verdict << "S1: bw=" << static_cast<int>(effective_bw) << "Hz"
            << " shape=" << shape_factor
            << " headroom=" << static_cast<int>(headroom) << "dB";

    ClassifyFeatures s1f;
    s1f.effective_bw = effective_bw;
    s1f.shape_factor = shape_factor;
    s1f.headroom_db = headroom;

    if (passed) {
        verdict << " PASS";
        return {SignalType::UNKNOWN, 0.5f, 1, 0.0f, true, verdict.str(), s1f};
    } else {
        verdict << " FAIL";
        return {SignalType::NOISE, 0.0f, 1, 0.0f, false, verdict.str(), s1f};
    }
}

// =========================================================================
// Stage 2: Amplitude Pattern (bimodality)
// =========================================================================
ClassifyResult CWClassifier::stage2_amplitude(const std::complex<float>* iq,
                                               int count, float fs) {
    // Envelope
    std::vector<float> env(count);
    for (int i = 0; i < count; i++) {
        env[i] = std::abs(iq[i]);
    }

    // Mean and stddev
    float mu = 0.0f;
    for (auto v : env) mu += v;
    mu /= count;

    float sigma = 0.0f;
    for (auto v : env) sigma += (v - mu) * (v - mu);
    sigma = std::sqrt(sigma / count);

    float bc, on_off;

    if (mu > 1e-6f && (sigma / mu) < 0.05f) {
        // Flat envelope → carrier
        bc = 0.0f;
        on_off = 1.0f;
    } else {
        if (sigma > 1e-6f) {
            float skew = 0.0f, kurt = 0.0f;
            for (auto v : env) {
                float z = (v - mu) / sigma;
                float z2 = z * z;
                skew += z * z2;
                kurt += z2 * z2;
            }
            skew /= count;
            kurt /= count;
            if (kurt < 1e-6f) kurt = 1e-6f;
            bc = (skew * skew + 1.0f) / kurt;
        } else {
            bc = 0.0f;
        }

        // On/off ratio
        int on_count = 0;
        for (auto v : env) {
            if (v > mu) on_count++;
        }
        on_off = static_cast<float>(on_count) / count;
    }

    std::ostringstream verdict;
    verdict << "S2: BC=" << bc << " on_off=" << on_off;

    // Fast exits
    if (on_off > 0.95f) {
        verdict << " CARRIER";
        return {SignalType::CARRIER, 0.0f, 2, 0.0f, false, verdict.str()};
    }
    if (bc < 0.20f) {
        verdict << " NOT_CW (BC low)";
        return {SignalType::NOISE, 0.0f, 2, 0.0f, false, verdict.str()};
    }

    // CW pass — relaxed for real-world HF fading
    bool pass_bc = bc > 0.30f;
    bool pass_on_off = on_off >= 0.10f && on_off <= 0.90f;

    ClassifyFeatures s2f;
    s2f.bimodality_coeff = bc;
    s2f.on_off_ratio = on_off;

    if (pass_bc && pass_on_off) {
        verdict << " PASS";
        return {SignalType::UNKNOWN, 0.6f, 2, 0.0f, true, verdict.str(), s2f};
    } else {
        // Don't hard-reject — pass with low confidence so Stage 3 can try
        verdict << " WEAK";
        return {SignalType::UNKNOWN, 0.3f, 2, 0.0f, true, verdict.str(), s2f};
    }
}

// =========================================================================
// Stage 3: Temporal Rhythm (autocorrelation)
// =========================================================================
ClassifyResult CWClassifier::stage3_rhythm(const std::complex<float>* iq,
                                            int count, float fs) {
    int N = count;

    // --- Carrier offset detection + removal ---
    // FFT to find carrier peak, then shift to DC
    int nfft = 1;
    while (nfft < N * 2) nfft <<= 1;

    fftwf_complex* cfft_in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfft);
    fftwf_complex* cfft_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfft);

    std::memset(cfft_in, 0, sizeof(fftwf_complex) * nfft);
    for (int i = 0; i < N; i++) {
        cfft_in[i][0] = iq[i].real();
        cfft_in[i][1] = iq[i].imag();
    }

    fftwf_plan carrier_plan = fftwf_plan_dft_1d(nfft, cfft_in, cfft_out,
                                                 FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(carrier_plan);
    fftwf_destroy_plan(carrier_plan);

    // Find peak bin
    int peak_idx = 0;
    float peak_mag = 0.0f;
    for (int i = 0; i < nfft; i++) {
        float mag = cfft_out[i][0]*cfft_out[i][0] + cfft_out[i][1]*cfft_out[i][1];
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_idx = i;
        }
    }

    // Parabolic interpolation
    float delta = 0.0f;
    if (peak_idx > 0 && peak_idx < nfft - 1) {
        float alpha = std::sqrt(cfft_out[peak_idx-1][0]*cfft_out[peak_idx-1][0]
                              + cfft_out[peak_idx-1][1]*cfft_out[peak_idx-1][1]);
        float beta  = std::sqrt(peak_mag);
        float gamma = std::sqrt(cfft_out[peak_idx+1][0]*cfft_out[peak_idx+1][0]
                              + cfft_out[peak_idx+1][1]*cfft_out[peak_idx+1][1]);
        float denom = alpha - 2.0f * beta + gamma;
        if (std::abs(denom) > 1e-10f) {
            delta = 0.5f * (alpha - gamma) / denom;
        }
    }
    float refined = peak_idx + delta;

    double carrier_offset;
    if (refined > nfft / 2.0f) {
        carrier_offset = (refined - nfft) * static_cast<double>(fs) / nfft;
    } else {
        carrier_offset = refined * static_cast<double>(fs) / nfft;
    }

    fftwf_free(cfft_in);
    fftwf_free(cfft_out);

    // Shift carrier to 0 Hz
    std::vector<std::complex<float>> shifted(N);
    for (int i = 0; i < N; i++) {
        double phase = -2.0 * M_PI * carrier_offset * i / fs;
        shifted[i] = iq[i] * std::complex<float>(std::cos(phase), std::sin(phase));
    }

    // 100 Hz lowpass (simple 2nd-order Butterworth via biquad)
    float fc = 100.0f / (fs / 2.0f);
    if (fc >= 1.0f) fc = 0.99f;
    float omega = std::tan(static_cast<float>(M_PI) * fc);
    float omega2 = omega * omega;
    float norm_bq = 1.0f / (1.0f + std::sqrt(2.0f) * omega + omega2);
    float b0 = omega2 * norm_bq;
    float b1 = 2.0f * b0;
    float b2 = b0;
    float a1 = 2.0f * (omega2 - 1.0f) * norm_bq;
    float a2 = (1.0f - std::sqrt(2.0f) * omega + omega2) * norm_bq;

    // Apply biquad to real and imag separately
    std::vector<std::complex<float>> filtered(N);
    float xr1 = 0, xr2 = 0, yr1 = 0, yr2 = 0;
    float xi1 = 0, xi2 = 0, yi1 = 0, yi2 = 0;
    for (int i = 0; i < N; i++) {
        float xr = shifted[i].real();
        float xi = shifted[i].imag();
        float yr = b0*xr + b1*xr1 + b2*xr2 - a1*yr1 - a2*yr2;
        float yi = b0*xi + b1*xi1 + b2*xi2 - a1*yi1 - a2*yi2;
        xr2 = xr1; xr1 = xr; yr2 = yr1; yr1 = yr;
        xi2 = xi1; xi1 = xi; yi2 = yi1; yi1 = yi;
        filtered[i] = {yr, yi};
    }

    // Envelope, mean-subtract
    std::vector<float> env(N);
    for (int i = 0; i < N; i++) {
        env[i] = std::abs(filtered[i]);
    }
    float env_mean = 0.0f;
    for (auto v : env) env_mean += v;
    env_mean /= N;
    for (auto& v : env) v -= env_mean;

    // --- Autocorrelation via Wiener-Khinchin: R = IFFT(|FFT(env)|²) ---
    // Use real-to-complex FFT for envelope
    int acorr_n = N;
    int acorr_nc = acorr_n / 2 + 1;

    float* acorr_in = (float*)fftwf_malloc(sizeof(float) * acorr_n);
    fftwf_complex* acorr_spec = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * acorr_nc);
    float* acorr_out = (float*)fftwf_malloc(sizeof(float) * acorr_n);

    std::memcpy(acorr_in, env.data(), sizeof(float) * acorr_n);

    fftwf_plan fwd = fftwf_plan_dft_r2c_1d(acorr_n, acorr_in, acorr_spec, FFTW_ESTIMATE);
    fftwf_execute(fwd);
    fftwf_destroy_plan(fwd);

    // |X[k]|²
    for (int i = 0; i < acorr_nc; i++) {
        float re = acorr_spec[i][0], im = acorr_spec[i][1];
        acorr_spec[i][0] = re * re + im * im;
        acorr_spec[i][1] = 0.0f;
    }

    fftwf_plan inv = fftwf_plan_dft_c2r_1d(acorr_n, acorr_spec, acorr_out, FFTW_ESTIMATE);
    fftwf_execute(inv);
    fftwf_destroy_plan(inv);

    // Normalize
    int R_len = acorr_n / 2;
    std::vector<float> R_norm(R_len);
    float R0 = acorr_out[0];
    if (R0 < 1e-12f) {
        fftwf_free(acorr_in);
        fftwf_free(acorr_spec);
        fftwf_free(acorr_out);
        return {SignalType::UNKNOWN, 0.0f, 3, 0.0f, false, "S3: zero-power signal"};
    }
    for (int i = 0; i < R_len; i++) {
        R_norm[i] = acorr_out[i] / R0;
    }

    fftwf_free(acorr_in);
    fftwf_free(acorr_spec);
    fftwf_free(acorr_out);

    // Adaptive threshold: 2/√N
    // Reduced from 5 to 2 based on live testing — real-world CW with fading
    // and QSB has rhythm scores of 0.05-0.15, much lower than synthetic signals.
    float threshold = 2.0f / std::sqrt(static_cast<float>(N));

    // --- RTTY sub-window: 38–55 baud ---
    int rtty_lo = std::max(2, static_cast<int>(fs / 55.0f));
    int rtty_hi = std::min(R_len - 1, static_cast<int>(fs / 38.0f));
    float rtty_score = 0.0f;
    int rtty_best_lag = 0;
    if (rtty_lo < rtty_hi && rtty_hi < R_len) {
        for (int i = rtty_lo; i <= rtty_hi; i++) {
            if (R_norm[i] > rtty_score) {
                // Check local max
                bool local_max = true;
                if (i > 0 && R_norm[i-1] > R_norm[i]) local_max = false;
                if (i < R_len-1 && R_norm[i+1] > R_norm[i]) local_max = false;
                if (local_max) {
                    rtty_score = R_norm[i];
                    rtty_best_lag = i;
                }
            }
        }
    }

    // --- CW search: dit period for WPM 3–50 ---
    // dit_samples = (1200/wpm/1000) * fs
    // Widen to 3 WPM to catch slow CW operators
    int lag_min = std::max(2, static_cast<int>((1200.0f / 50.0f / 1000.0f) * fs));
    int lag_max = std::min(R_len - 2, static_cast<int>((1200.0f / 3.0f / 1000.0f) * fs) + 1);

    if (lag_min >= lag_max || lag_max >= R_len) {
        return {SignalType::UNKNOWN, 0.0f, 3, 0.0f, false, "S3: search window invalid"};
    }

    // CW on/off keying autocorrelation structure:
    //   - NEGATIVE trough at dit period (on aligns with off → anti-correlation)
    //   - POSITIVE peak at 2*dit period (on aligns with on again)
    //
    // Strategy: Search for both, prefer the one that gives a definitive signal.
    // A strong negative trough at lag L means dit period = L, WPM = 1200/(L/fs*1000)
    // A strong positive peak at lag L means dit period = L/2, WPM = 1200/(L/2/fs*1000)

    // 1) Search for positive peaks (first above threshold)
    float pos_score = 0.0f;
    int pos_lag = 0;
    for (int i = lag_min; i <= lag_max; i++) {
        if (i > 0 && i < R_len - 1 &&
            R_norm[i] > R_norm[i-1] && R_norm[i] >= R_norm[i+1] &&
            R_norm[i] > threshold) {
            pos_score = R_norm[i];
            pos_lag = i;
            break;
        }
    }

    // 2) Search for negative troughs (first below -threshold)
    float neg_score = 0.0f;
    int neg_lag = 0;
    for (int i = lag_min; i <= lag_max; i++) {
        if (i > 0 && i < R_len - 1 &&
            R_norm[i] < R_norm[i-1] && R_norm[i] <= R_norm[i+1] &&
            R_norm[i] < -threshold) {
            neg_score = std::abs(R_norm[i]);
            neg_lag = i;
            break;
        }
    }

    // Choose the best indicator
    float cw_score = 0.0f;
    int best_lag = 0;
    bool from_trough = false;

    if (neg_lag > 0 && pos_lag > 0) {
        // Both found — prefer whichever is stronger
        if (neg_score > pos_score) {
            cw_score = neg_score;
            best_lag = neg_lag;
            from_trough = true;
        } else {
            cw_score = pos_score;
            best_lag = pos_lag;
        }
    } else if (neg_lag > 0) {
        cw_score = neg_score;
        best_lag = neg_lag;
        from_trough = true;
    } else if (pos_lag > 0) {
        cw_score = pos_score;
        best_lag = pos_lag;
    }

    // 3) Fallback: strongest positive peak for reporting
    if (best_lag == 0) {
        for (int i = lag_min; i <= lag_max; i++) {
            if (i > 0 && i < R_len - 1 &&
                R_norm[i] > R_norm[i-1] && R_norm[i] >= R_norm[i+1] &&
                R_norm[i] > cw_score) {
                cw_score = R_norm[i];
                best_lag = i;
            }
        }
    }
    // Also check strongest negative trough as fallback
    if (best_lag == 0) {
        for (int i = lag_min; i <= lag_max; i++) {
            if (i > 0 && i < R_len - 1 &&
                R_norm[i] < R_norm[i-1] && R_norm[i] <= R_norm[i+1] &&
                std::abs(R_norm[i]) > cw_score) {
                cw_score = std::abs(R_norm[i]);
                best_lag = i;
                from_trough = true;
            }
        }
    }

    // RTTY vs CW decision
    bool rtty_wins = false;
    if (rtty_score > threshold && rtty_best_lag > 0) {
        // CW peak is harmonic of RTTY?
        if (best_lag > 0 && std::abs(best_lag - 2 * rtty_best_lag) <= std::max(1, static_cast<int>(rtty_best_lag * 0.15))) {
            rtty_wins = true;
        } else if (rtty_score > cw_score) {
            rtty_wins = true;
        }
    }

    if (rtty_wins) {
        float period_ms = (rtty_best_lag / fs) * 1000.0f;
        float baud = (period_ms > 0) ? 1000.0f / period_ms : 0.0f;
        std::ostringstream v;
        v << "S3: RTTY baud=" << baud << " score=" << rtty_score;
        return {SignalType::RTTY, 0.0f, 3, 0.0f, false, v.str()};
    }

    if (best_lag == 0) {
        return {SignalType::UNKNOWN, 0.0f, 3, 0.0f, false,
                "S3: no autocorrelation peak in CW range"};
    }

    float peak_lag_ms = (best_lag / fs) * 1000.0f;
    // If from trough: lag = dit period directly
    // If from positive peak: lag = 2*dit period, so dit = lag/2
    float dit_ms = from_trough ? peak_lag_ms : peak_lag_ms / 2.0f;
    float wpm_est = (dit_ms > 0) ? 1200.0f / dit_ms : 0.0f;

    std::ostringstream verdict;
    verdict << "S3: lag=" << best_lag << "samp (" << peak_lag_ms << "ms)"
            << " score=" << cw_score
            << " thresh=" << threshold
            << " wpm=" << wpm_est;

    ClassifyFeatures s3f;
    s3f.rhythm_score = cw_score;

    if (cw_score > threshold) {
        verdict << " PASS";
        return {SignalType::CW, 0.7f, 3, wpm_est, true, verdict.str(), s3f};
    } else if (cw_score > threshold * 0.5f) {
        // Close enough — mark as possible CW with reduced confidence
        verdict << " MARGINAL";
        return {SignalType::CW, cw_score / threshold * 0.5f, 3, wpm_est, true, verdict.str(), s3f};
    } else {
        verdict << " FAIL";
        return {SignalType::UNKNOWN, cw_score / threshold * 0.3f, 3, wpm_est, false, verdict.str(), s3f};
    }
}

// =========================================================================
// Helper: Bandwidth calculation
// =========================================================================
float CWClassifier::calc_bandwidth(const float* psd_db, const float* freqs,
                                    int nbins, float peak_db, float threshold_db) {
    // Find the peak bin
    int max_idx = 0;
    for (int i = 1; i < nbins; i++) {
        if (psd_db[i] > psd_db[max_idx]) max_idx = i;
    }

    if (psd_db[max_idx] < threshold_db) return 0.0f;

    // Walk left
    int left = max_idx;
    while (left > 0 && psd_db[left - 1] >= threshold_db) left--;

    // Walk right
    int right = max_idx;
    while (right < nbins - 1 && psd_db[right + 1] >= threshold_db) right++;

    return freqs[right] - freqs[left];
}
