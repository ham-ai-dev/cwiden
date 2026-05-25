#pragma once
/**
 * tui.hpp — FTXUI terminal user interface for cwiden.
 *
 * Supports two modes:
 *   - Normal: displays active CW signals + SDR config + band selector
 *   - Validation: interactive signal selection, audio playback, Y/N labeling
 */

#include <string>
#include <vector>
#include <mutex>
#include <functional>
#include <atomic>
#include <cstdint>

#include <ftxui/component/screen_interactive.hpp>

struct TrackedSignal;  // forward decl

class Tui {
public:
    Tui();

    /** Main event loop — blocks until user quits (q or Ctrl+C). */
    void run();

    // Thread-safe setters called from the pipeline thread
    static void update_signals(const std::vector<TrackedSignal>& signals);
    static void update_spectrum(const std::vector<float>& psd_db, int nbins,
                                double center_freq, double freq_per_bin);
    static void update_sdr_info(const std::string& device, double freq_hz,
                                uint32_t rate, bool connected);
    static void set_channel_count(int count);
    static void set_uptime(double seconds);
    static void set_status_message(const std::string& msg);

    // Config change callback
    using ConfigChangeCallback = std::function<void(const std::string&, const std::string&)>;
    static void set_config_change_callback(ConfigChangeCallback cb);
    static void set_initial_config(double center_freq, uint32_t lna_gain,
                                   uint32_t vga_gain, const std::string& band);

    // Band selector
    struct BandInfo {
        std::string name;
        double center_hz;
        double cw_lo_hz;
        double cw_hi_hz;
    };
    using BandChangeCallback = std::function<void(int band_index)>;
    static void set_bands(const std::vector<BandInfo>& bands);
    static void set_band_change_callback(BandChangeCallback cb);
    static void set_current_band(int index);

    // CW filter range — only show signals within this range
    static void set_cw_filter(double lo_hz, double hi_hz);

    // Validation mode
    using ValidationCallback = std::function<void(double freq_hz, int label)>;
    using AudioSelectCallback = std::function<void(int signal_idx, double freq_hz)>;
    using DeviceSwitchCallback = std::function<bool(const std::string& device_name)>;
    using TestToneCallback = std::function<void()>;
    static void set_validation_callback(ValidationCallback cb);
    static void set_audio_select_callback(AudioSelectCallback cb);
    static void set_device_switch_callback(DeviceSwitchCallback cb);
    static void set_test_tone_callback(TestToneCallback cb);
    static void set_validation_mode(bool enabled);

    // Scan mode: manual frequency tuning
    using ScanPlayCallback = std::function<void(double freq_hz)>;
    static void set_scan_play_callback(ScanPlayCallback cb);

    // Audio device list
    struct DeviceInfo {
        std::string name;
        std::string nick;
    };
    static void set_audio_devices(const std::vector<DeviceInfo>& devices);

private:
    ftxui::ScreenInteractive screen_;

    // Shared state (protected by mutex)
    static std::mutex                  mutex_;
    static std::vector<TrackedSignal>  signals_;
    static std::vector<float>          spectrum_db_;
    static int                         spectrum_bins_;
    static double                      spectrum_center_;
    static double                      spectrum_bin_hz_;
    static std::string                 sdr_device_;
    static double                      sdr_freq_;
    static uint32_t                    sdr_rate_;
    static bool                        sdr_connected_;
    static int                         channel_count_;
    static double                      uptime_s_;
    static ConfigChangeCallback        config_cb_;

    // Initial config
    static std::string init_freq_str_;
    static std::string init_lna_str_;
    static std::string init_vga_str_;
    static std::string init_band_str_;

    // Band selector
    static std::vector<BandInfo>       bands_;
    static int                         selected_band_;
    static BandChangeCallback          band_change_cb_;
    static double                      cw_filter_lo_;
    static double                      cw_filter_hi_;

    // Validation mode
    static ValidationCallback          validation_cb_;
    static AudioSelectCallback         audio_select_cb_;
    static DeviceSwitchCallback        device_switch_cb_;
    static TestToneCallback            test_tone_cb_;
    static int                         selected_signal_;
    static int                         playing_signal_;
    static std::string                 status_msg_;
    static bool                        validation_mode_;
    static int                         labels_positive_;
    static int                         labels_negative_;

    // Audio device picker
    static std::vector<DeviceInfo>     audio_devices_;
    static int                         selected_device_;

    // Focus: 0=signals/scan, 1=bands, 2=audio devices
    static int                         focus_panel_;

    // Scan mode
    static bool                        scan_mode_;        // toggle with 'F'
    static double                      scan_freq_;        // current dial frequency
    static double                      scan_step_;        // Hz per arrow press
    static ScanPlayCallback            scan_play_cb_;
    static int                         scan_labels_cw_;
    static int                         scan_labels_noise_;
};
