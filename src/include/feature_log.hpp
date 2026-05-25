#pragma once
/**
 * feature_log.hpp — Log classifier features + user labels for threshold tuning.
 */

#include <string>
#include <vector>
#include <fstream>
#include <mutex>

struct FeatureRow {
    double timestamp;
    double freq_hz;
    float  snr_db;
    // Stage 1
    float  effective_bw;
    float  shape_factor;
    float  headroom_db;
    // Stage 2
    float  bimodality_coeff;
    float  on_off_ratio;
    // Stage 3
    float  rhythm_score;
    float  wpm_estimate;
    // ML features
    float  spectral_entropy;
    float  peak_stability;
    // Classifier output
    float  confidence;
    bool   classified_cw;
    int    stage_reached;
    // User label (-1 = unlabeled, 0 = not CW, 1 = CW)
    int    user_label = -1;
};

class FeatureLog {
public:
    FeatureLog();
    ~FeatureLog();

    /** Open log file for writing. Creates header if new. */
    bool open(const std::string& path);

    /** Append a feature row. */
    void log(const FeatureRow& row);

    /** Read all rows from a log file (for tuning). */
    static std::vector<FeatureRow> load(const std::string& path);

    /** Update the label of the most recent row matching freq_hz (within 100 Hz). */
    void update_label(double freq_hz, int label);

    /** Get path of current log. */
    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::ofstream file_;
    std::mutex mutex_;

    // In-memory copy for label updates
    std::vector<FeatureRow> rows_;
};

/**
 * Compute optimal thresholds from labeled data.
 * Returns a map of parameter name → suggested threshold.
 */
struct TunedThresholds {
    float bc_min = 0.40f;
    float on_off_min = 0.15f;
    float on_off_max = 0.85f;
    float rhythm_threshold_mult = 2.0f;
    float min_snr = 3.0f;
    float min_confidence = 0.3f;

    void print() const;
    bool save(const std::string& path) const;
    static TunedThresholds load(const std::string& path);
};

TunedThresholds compute_thresholds(const std::vector<FeatureRow>& data);
