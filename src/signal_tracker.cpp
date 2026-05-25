#include "signal_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <ctime>

std::string TrackedSignal::status_str() const {
    switch (state) {
        case TrackState::NEW:       return "NEW";
        case TrackState::TENTATIVE: return "TENTATIVE";
        case TrackState::ACTIVE:    return "ACTIVE";
        case TrackState::STALE:     return "STALE";
    }
    return "?";
}

SignalTracker::SignalTracker(int confirm_count, double stale_timeout, double merge_hz)
    : confirm_count_(confirm_count), stale_timeout_(stale_timeout), merge_hz_(merge_hz) {}

int SignalTracker::find_nearest(double freq_hz) const {
    int best = -1;
    double best_dist = merge_hz_ + 1.0;
    for (int i = 0; i < static_cast<int>(signals_.size()); i++) {
        double dist = std::abs(signals_[i].freq_hz - freq_hz);
        if (dist < merge_hz_ && dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

void SignalTracker::update(
        const std::vector<std::tuple<double, float, float, float>>& detections,
        double now) {
    // Mark all EXISTING signals as "not seen this cycle"
    // New signals added below will not be in this vector.
    int original_count = static_cast<int>(signals_.size());
    std::vector<bool> matched(original_count, false);

    for (auto& [freq, snr, wpm, conf] : detections) {
        int idx = find_nearest(freq);
        if (idx >= 0) {
            // Update existing
            auto& s = signals_[idx];
            s.freq_hz = 0.9 * s.freq_hz + 0.1 * freq;  // smooth freq
            s.snr_db = snr;
            s.wpm = 0.8f * s.wpm + 0.2f * wpm;  // smooth WPM
            s.confidence = conf;
            s.consecutive_hits++;
            s.consecutive_misses = 0;
            s.last_seen = now;
            matched[idx] = true;

            // State transitions
            if (s.consecutive_hits >= confirm_count_) {
                s.state = TrackState::ACTIVE;
            } else if (s.consecutive_hits >= 1) {
                s.state = TrackState::TENTATIVE;
            }
        } else {
            // New signal
            TrackedSignal s;
            s.freq_hz = freq;
            s.snr_db = snr;
            s.wpm = wpm;
            s.confidence = conf;
            s.state = TrackState::NEW;
            s.consecutive_hits = 1;
            s.consecutive_misses = 0;
            s.first_seen = now;
            s.last_seen = now;
            signals_.push_back(s);
        }
    }

    // Increment misses for unmatched signals (only the original ones)
    for (int i = 0; i < original_count; i++) {
        if (!matched[i]) {
            signals_[i].consecutive_misses++;
            signals_[i].consecutive_hits = 0;

            if (now - signals_[i].last_seen > stale_timeout_) {
                signals_[i].state = TrackState::STALE;
            }
        }
    }

    // Remove stale signals
    signals_.erase(
        std::remove_if(signals_.begin(), signals_.end(),
            [&](const TrackedSignal& s) {
                return s.state == TrackState::STALE &&
                       now - s.last_seen > stale_timeout_ * 2.0;
            }),
        signals_.end());

    // Sort by frequency
    std::sort(signals_.begin(), signals_.end(),
        [](const TrackedSignal& a, const TrackedSignal& b) {
            return a.freq_hz < b.freq_hz;
        });
}

std::vector<const TrackedSignal*> SignalTracker::active_signals() const {
    std::vector<const TrackedSignal*> result;
    for (auto& s : signals_) {
        if (s.state == TrackState::ACTIVE) result.push_back(&s);
    }
    return result;
}

std::vector<const TrackedSignal*> SignalTracker::visible_signals() const {
    std::vector<const TrackedSignal*> result;
    for (auto& s : signals_) {
        if (s.state == TrackState::TENTATIVE || s.state == TrackState::ACTIVE) {
            result.push_back(&s);
        }
    }
    return result;
}

std::string SignalTracker::format_text(double now) const {
    auto vis = visible_signals();
    if (vis.empty()) return "";

    std::ostringstream out;
    time_t t = time(nullptr);
    struct tm* tm = localtime(&t);
    char tbuf[16];
    strftime(tbuf, sizeof(tbuf), "%H:%M:%S", tm);

    for (auto* s : vis) {
        char line[128];
        snprintf(line, sizeof(line),
                 "[%s] CW %10.3f kHz  SNR=%2.0fdB  WPM≈%2.0f  conf=%.2f  %s",
                 tbuf,
                 s->freq_hz / 1000.0,
                 s->snr_db,
                 s->wpm,
                 s->confidence,
                 s->status_str().c_str());
        out << line << "\n";
    }
    return out.str();
}

std::string SignalTracker::format_json(double now) const {
    auto vis = visible_signals();

    std::ostringstream out;
    out << "{\"ts\":\"";

    time_t t = time(nullptr);
    struct tm* tm = localtime(&t);
    char tbuf[32];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", tm);
    out << tbuf << "\",\"signals\":[";

    for (size_t i = 0; i < vis.size(); i++) {
        auto* s = vis[i];
        if (i > 0) out << ",";
        out << "{\"freq_hz\":" << static_cast<long long>(s->freq_hz)
            << ",\"snr_db\":" << std::fixed << std::setprecision(1) << s->snr_db
            << ",\"wpm\":" << static_cast<int>(s->wpm)
            << ",\"confidence\":" << std::setprecision(2) << s->confidence
            << ",\"status\":\"" << s->status_str() << "\""
            << ",\"age_s\":" << static_cast<int>(s->age(now)) << "}";
    }
    out << "]}";
    return out.str();
}
