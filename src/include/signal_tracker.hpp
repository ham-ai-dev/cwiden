#pragma once
/**
 * signal_tracker.hpp — Per-frequency state machine for CW signal persistence.
 *
 * Tracks detected CW signals across classification cycles and applies
 * persistence filtering to suppress transient false positives.
 *
 * State machine: NEW → TENTATIVE → ACTIVE → STALE → (removed)
 */

#include <vector>
#include <string>
#include <chrono>
#include <map>

enum class TrackState {
    NEW,        // First detection, needs confirmation
    TENTATIVE,  // 1-2 consecutive positives
    ACTIVE,     // 3+ consecutive positives — confirmed CW
    STALE       // No positive for > timeout — pending removal
};

struct TrackedSignal {
    double     freq_hz;
    float      snr_db;
    float      wpm;
    float      confidence;
    TrackState state;
    int        consecutive_hits;
    int        consecutive_misses;
    double     first_seen;   // seconds since start
    double     last_seen;    // seconds since start

    std::string status_str() const;
    double age(double now) const { return now - first_seen; }
};

class SignalTracker {
public:
    /**
     * @param confirm_count  Number of consecutive hits to reach ACTIVE
     * @param stale_timeout  Seconds without a hit before marking STALE
     * @param merge_hz       Frequency merge window (same signal drift)
     */
    SignalTracker(int confirm_count = 3, double stale_timeout = 10.0,
                  double merge_hz = 50.0);

    /**
     * Update tracker with a new set of classified signals.
     * Call once per classification cycle.
     *
     * @param detections  Vector of (freq_hz, snr_db, wpm, confidence) tuples
     *                    — only signals that passed CW classification.
     * @param now         Current time in seconds since start.
     */
    void update(const std::vector<std::tuple<double, float, float, float>>& detections,
                double now);

    /** Get all currently tracked signals (any state). */
    const std::vector<TrackedSignal>& signals() const { return signals_; }

    /** Get only ACTIVE signals. */
    std::vector<const TrackedSignal*> active_signals() const;

    /** Get all signals that should be displayed (TENTATIVE + ACTIVE). */
    std::vector<const TrackedSignal*> visible_signals() const;

    /** Format signals for headless text output. */
    std::string format_text(double now) const;

    /** Format signals for headless JSON output. */
    std::string format_json(double now) const;

private:
    int    confirm_count_;
    double stale_timeout_;
    double merge_hz_;

    std::vector<TrackedSignal> signals_;

    // Find existing signal within merge distance, or -1
    int find_nearest(double freq_hz) const;
};
