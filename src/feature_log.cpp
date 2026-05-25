#include "feature_log.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

FeatureLog::FeatureLog() {}
FeatureLog::~FeatureLog() {
    if (file_.is_open()) file_.close();
}

bool FeatureLog::open(const std::string& path) {
    path_ = path;

    // Check if file exists already
    bool exists = false;
    {
        std::ifstream check(path);
        exists = check.good();
    }

    file_.open(path, std::ios::app);
    if (!file_.is_open()) {
        std::cerr << "Cannot open feature log: " << path << std::endl;
        return false;
    }

    if (!exists) {
        file_ << "timestamp,freq_hz,snr_db,eff_bw,shape_factor,headroom_db,"
              << "bc,on_off,rhythm_score,wpm,confidence,classified_cw,"
              << "stage_reached,user_label" << std::endl;
    }

    return true;
}

void FeatureLog::log(const FeatureRow& row) {
    std::lock_guard<std::mutex> lock(mutex_);

    rows_.push_back(row);

    if (file_.is_open()) {
        file_ << std::fixed << std::setprecision(3)
              << row.timestamp << ","
              << static_cast<long long>(row.freq_hz) << ","
              << row.snr_db << ","
              << row.effective_bw << ","
              << row.shape_factor << ","
              << row.headroom_db << ","
              << row.bimodality_coeff << ","
              << row.on_off_ratio << ","
              << row.rhythm_score << ","
              << row.wpm_estimate << ","
              << row.confidence << ","
              << (row.classified_cw ? 1 : 0) << ","
              << row.stage_reached << ","
              << row.user_label
              << std::endl;
    }
}

void FeatureLog::update_label(double freq_hz, int label) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find most recent row near this frequency
    for (int i = static_cast<int>(rows_.size()) - 1; i >= 0; i--) {
        if (std::abs(rows_[i].freq_hz - freq_hz) < 100.0) {
            rows_[i].user_label = label;

            // Rewrite the file with updated labels
            if (file_.is_open()) {
                file_.close();
                file_.open(path_, std::ios::trunc);
                file_ << "timestamp,freq_hz,snr_db,eff_bw,shape_factor,headroom_db,"
                      << "bc,on_off,rhythm_score,wpm,confidence,classified_cw,"
                      << "stage_reached,user_label" << std::endl;
                for (auto& r : rows_) {
                    file_ << std::fixed << std::setprecision(3)
                          << r.timestamp << ","
                          << static_cast<long long>(r.freq_hz) << ","
                          << r.snr_db << ","
                          << r.effective_bw << ","
                          << r.shape_factor << ","
                          << r.headroom_db << ","
                          << r.bimodality_coeff << ","
                          << r.on_off_ratio << ","
                          << r.rhythm_score << ","
                          << r.wpm_estimate << ","
                          << r.confidence << ","
                          << (r.classified_cw ? 1 : 0) << ","
                          << r.stage_reached << ","
                          << r.user_label
                          << std::endl;
                }
            }
            return;
        }
    }
}

std::vector<FeatureRow> FeatureLog::load(const std::string& path) {
    std::vector<FeatureRow> rows;
    std::ifstream file(path);
    if (!file.is_open()) return rows;

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        FeatureRow r;
        std::istringstream ss(line);
        std::string token;
        int col = 0;
        while (std::getline(ss, token, ',')) {
            try {
                switch (col) {
                    case 0: r.timestamp = std::stod(token); break;
                    case 1: r.freq_hz = std::stod(token); break;
                    case 2: r.snr_db = std::stof(token); break;
                    case 3: r.effective_bw = std::stof(token); break;
                    case 4: r.shape_factor = std::stof(token); break;
                    case 5: r.headroom_db = std::stof(token); break;
                    case 6: r.bimodality_coeff = std::stof(token); break;
                    case 7: r.on_off_ratio = std::stof(token); break;
                    case 8: r.rhythm_score = std::stof(token); break;
                    case 9: r.wpm_estimate = std::stof(token); break;
                    case 10: r.confidence = std::stof(token); break;
                    case 11: r.classified_cw = (std::stoi(token) != 0); break;
                    case 12: r.stage_reached = std::stoi(token); break;
                    case 13: r.user_label = std::stoi(token); break;
                }
            } catch (...) {}
            col++;
        }
        rows.push_back(r);
    }
    return rows;
}

// =========================================================================
// Threshold tuning
// =========================================================================

void TunedThresholds::print() const {
    std::cerr << "=== Tuned Thresholds ===" << std::endl;
    std::cerr << "  bc_min:              " << bc_min << std::endl;
    std::cerr << "  on_off_min:          " << on_off_min << std::endl;
    std::cerr << "  on_off_max:          " << on_off_max << std::endl;
    std::cerr << "  rhythm_thresh_mult:  " << rhythm_threshold_mult << std::endl;
    std::cerr << "  min_snr:             " << min_snr << std::endl;
    std::cerr << "  min_confidence:      " << min_confidence << std::endl;
}

bool TunedThresholds::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "{" << std::endl;
    f << "  \"bc_min\": " << bc_min << "," << std::endl;
    f << "  \"on_off_min\": " << on_off_min << "," << std::endl;
    f << "  \"on_off_max\": " << on_off_max << "," << std::endl;
    f << "  \"rhythm_threshold_mult\": " << rhythm_threshold_mult << "," << std::endl;
    f << "  \"min_snr\": " << min_snr << "," << std::endl;
    f << "  \"min_confidence\": " << min_confidence << std::endl;
    f << "}" << std::endl;
    return true;
}

TunedThresholds TunedThresholds::load(const std::string& path) {
    TunedThresholds t;
    std::ifstream f(path);
    if (!f.is_open()) return t;

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    auto extract = [&](const std::string& key) -> float {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return -1;
        pos = content.find(":", pos);
        if (pos == std::string::npos) return -1;
        return std::stof(content.substr(pos + 1));
    };

    float v;
    if ((v = extract("bc_min")) >= 0) t.bc_min = v;
    if ((v = extract("on_off_min")) >= 0) t.on_off_min = v;
    if ((v = extract("on_off_max")) >= 0) t.on_off_max = v;
    if ((v = extract("rhythm_threshold_mult")) >= 0) t.rhythm_threshold_mult = v;
    if ((v = extract("min_snr")) >= 0) t.min_snr = v;
    if ((v = extract("min_confidence")) >= 0) t.min_confidence = v;

    return t;
}

TunedThresholds compute_thresholds(const std::vector<FeatureRow>& data) {
    TunedThresholds t;

    // Split into positive (CW) and negative (not CW)
    std::vector<const FeatureRow*> pos, neg;
    for (auto& r : data) {
        if (r.user_label == 1) pos.push_back(&r);
        else if (r.user_label == 0) neg.push_back(&r);
    }

    if (pos.empty() || neg.empty()) {
        std::cerr << "Need both positive and negative labels to tune." << std::endl;
        std::cerr << "  Positive (CW): " << pos.size() << std::endl;
        std::cerr << "  Negative (not CW): " << neg.size() << std::endl;
        return t;
    }

    std::cerr << "Tuning from " << pos.size() << " CW and "
              << neg.size() << " non-CW samples" << std::endl;

    // --- BC threshold: find value that best separates ---
    // Sort all BC values, find the split point
    auto find_best_threshold = [](const std::vector<const FeatureRow*>& pos,
                                   const std::vector<const FeatureRow*>& neg,
                                   auto getter, float lo, float hi, bool higher_is_pos) -> float {
        float best_thresh = (lo + hi) / 2;
        float best_score = 0;
        int steps = 50;
        for (int i = 0; i <= steps; i++) {
            float thresh = lo + (hi - lo) * i / steps;
            int tp = 0, tn = 0;
            for (auto* r : pos) {
                float v = getter(r);
                if (higher_is_pos ? (v >= thresh) : (v <= thresh)) tp++;
            }
            for (auto* r : neg) {
                float v = getter(r);
                if (higher_is_pos ? (v < thresh) : (v > thresh)) tn++;
            }
            float score = static_cast<float>(tp + tn) / (pos.size() + neg.size());
            if (score > best_score) {
                best_score = score;
                best_thresh = thresh;
            }
        }
        return best_thresh;
    };

    t.bc_min = find_best_threshold(pos, neg,
        [](const FeatureRow* r) { return r->bimodality_coeff; },
        0.1f, 0.8f, true);

    t.on_off_min = find_best_threshold(pos, neg,
        [](const FeatureRow* r) { return r->on_off_ratio; },
        0.05f, 0.45f, true);

    // Rhythm score: find the multiplier that works best
    // rhythm_threshold = mult / sqrt(N), but we don't know N, so tune on raw score
    float best_rhythm_thresh = find_best_threshold(pos, neg,
        [](const FeatureRow* r) { return r->rhythm_score; },
        0.01f, 0.3f, true);

    // Estimate equivalent multiplier assuming ~400 samples
    t.rhythm_threshold_mult = best_rhythm_thresh * std::sqrt(400.0f);

    // SNR
    t.min_snr = find_best_threshold(pos, neg,
        [](const FeatureRow* r) { return r->snr_db; },
        1.0f, 15.0f, true);

    // Confidence
    t.min_confidence = find_best_threshold(pos, neg,
        [](const FeatureRow* r) { return r->confidence; },
        0.1f, 0.7f, true);

    return t;
}
