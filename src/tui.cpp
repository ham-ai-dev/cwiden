#include "tui.hpp"
#include "signal_tracker.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <thread>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace ftxui;

// Static member initialization
std::mutex                  Tui::mutex_;
std::vector<TrackedSignal>  Tui::signals_;
std::vector<float>          Tui::spectrum_db_;
int                         Tui::spectrum_bins_ = 0;
double                      Tui::spectrum_center_ = 0.0;
double                      Tui::spectrum_bin_hz_ = 0.0;
std::string                 Tui::sdr_device_ = "HackRF";
double                      Tui::sdr_freq_ = 0.0;
uint32_t                    Tui::sdr_rate_ = 0;
bool                        Tui::sdr_connected_ = false;
int                         Tui::channel_count_ = 0;
double                      Tui::uptime_s_ = 0.0;
Tui::ConfigChangeCallback   Tui::config_cb_;
std::string                 Tui::init_freq_str_ = "6900000";
std::string                 Tui::init_lna_str_ = "16";
std::string                 Tui::init_vga_str_ = "20";
std::string                 Tui::init_band_str_ = "40m";

// Band selector
std::vector<Tui::BandInfo>  Tui::bands_;
int                         Tui::selected_band_ = 2; // default 40m
Tui::BandChangeCallback     Tui::band_change_cb_;
double                      Tui::cw_filter_lo_ = 0.0;
double                      Tui::cw_filter_hi_ = 1e12;

// Validation mode statics
Tui::ValidationCallback     Tui::validation_cb_;
Tui::AudioSelectCallback    Tui::audio_select_cb_;
Tui::DeviceSwitchCallback   Tui::device_switch_cb_;
Tui::TestToneCallback       Tui::test_tone_cb_;
int                         Tui::selected_signal_ = 0;
int                         Tui::playing_signal_ = -1;
std::string                 Tui::status_msg_ = "Select a band to start";
bool                        Tui::validation_mode_ = false;
int                         Tui::labels_positive_ = 0;
int                         Tui::labels_negative_ = 0;
int                         Tui::labels_unknown_ = 0;
std::vector<Tui::DeviceInfo> Tui::audio_devices_;
int                         Tui::selected_device_ = 0;
int                         Tui::focus_panel_ = 1; // start on bands

// Scan mode
bool                        Tui::scan_mode_ = false;
double                      Tui::scan_freq_ = 7030000;
double                      Tui::scan_step_ = 200;     // 200 Hz steps
Tui::ScanPlayCallback       Tui::scan_play_cb_;
int                         Tui::scan_labels_cw_ = 0;
int                         Tui::scan_labels_noise_ = 0;
int                         Tui::scan_labels_unknown_ = 0;

static std::string format_freq(double hz) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", hz / 1000.0);
    return buf;
}

static std::string format_uptime(double s) {
    int mins = static_cast<int>(s) / 60;
    int secs = static_cast<int>(s) % 60;
    char buf[16];
    snprintf(buf, sizeof(buf), "%dm%02ds", mins, secs);
    return buf;
}

Tui::Tui() : screen_(ScreenInteractive::Fullscreen()) {}

void Tui::set_initial_config(double center_freq, uint32_t lna_gain,
                              uint32_t vga_gain, const std::string& band) {
    init_freq_str_ = std::to_string(static_cast<long long>(center_freq));
    init_lna_str_ = std::to_string(lna_gain);
    init_vga_str_ = std::to_string(vga_gain);
    init_band_str_ = band;
}

void Tui::update_signals(const std::vector<TrackedSignal>& signals) {
    std::lock_guard<std::mutex> lock(mutex_);
    signals_ = signals;
}

void Tui::update_spectrum(const std::vector<float>& psd_db, int nbins,
                           double center_freq, double freq_per_bin) {
    std::lock_guard<std::mutex> lock(mutex_);
    spectrum_db_ = psd_db;
    spectrum_bins_ = nbins;
    spectrum_center_ = center_freq;
    spectrum_bin_hz_ = freq_per_bin;
}

void Tui::update_sdr_info(const std::string& device, double freq_hz,
                           uint32_t rate, bool connected) {
    std::lock_guard<std::mutex> lock(mutex_);
    sdr_device_ = device;
    sdr_freq_ = freq_hz;
    sdr_rate_ = rate;
    sdr_connected_ = connected;
}

void Tui::set_channel_count(int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    channel_count_ = count;
}

void Tui::set_uptime(double seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    uptime_s_ = seconds;
}

void Tui::set_config_change_callback(ConfigChangeCallback cb) { config_cb_ = cb; }
void Tui::set_validation_callback(ValidationCallback cb) { validation_cb_ = cb; }
void Tui::set_audio_select_callback(AudioSelectCallback cb) { audio_select_cb_ = cb; }
void Tui::set_device_switch_callback(DeviceSwitchCallback cb) { device_switch_cb_ = cb; }
void Tui::set_test_tone_callback(TestToneCallback cb) { test_tone_cb_ = cb; }
void Tui::set_validation_mode(bool enabled) { validation_mode_ = enabled; }
void Tui::set_scan_play_callback(ScanPlayCallback cb) { scan_play_cb_ = cb; }
void Tui::set_audio_devices(const std::vector<DeviceInfo>& devices) {
    std::lock_guard<std::mutex> lock(mutex_);
    audio_devices_ = devices;
}
void Tui::set_status_message(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_msg_ = msg;
}
void Tui::set_bands(const std::vector<BandInfo>& bands) {
    std::lock_guard<std::mutex> lock(mutex_);
    bands_ = bands;
}
void Tui::set_band_change_callback(BandChangeCallback cb) { band_change_cb_ = cb; }
void Tui::set_current_band(int index) {
    std::lock_guard<std::mutex> lock(mutex_);
    selected_band_ = index;
    if (index >= 0 && index < static_cast<int>(bands_.size())) {
        cw_filter_lo_ = bands_[index].cw_lo_hz;
        cw_filter_hi_ = bands_[index].cw_hi_hz;
    }
}
void Tui::set_cw_filter(double lo_hz, double hi_hz) {
    std::lock_guard<std::mutex> lock(mutex_);
    cw_filter_lo_ = lo_hz;
    cw_filter_hi_ = hi_hz;
}

void Tui::run() {
    std::string freq_str = init_freq_str_;
    std::string lna_str = init_lna_str_;
    std::string vga_str = init_vga_str_;

    auto freq_input = Input(&freq_str, "Center Freq Hz");
    auto lna_input = Input(&lna_str, "LNA dB");
    auto vga_input = Input(&vga_str, "VGA dB");

    auto apply_btn = Button("Apply", [&] {
        if (config_cb_) {
            config_cb_("center_freq", freq_str);
            config_cb_("lna_gain", lna_str);
            config_cb_("vga_gain", vga_str);
            status_msg_ = "Settings applied";
        }
    });

    auto main_container = Container::Vertical({
        freq_input, lna_input, vga_input, apply_btn,
    });

    auto renderer = Renderer(main_container, [&] {
        std::lock_guard<std::mutex> lock(mutex_);

        // Build visible signals list — FILTERED to CW sub-band only
        std::vector<int> visible_indices;
        for (int i = 0; i < static_cast<int>(signals_.size()); i++) {
            if ((signals_[i].state == TrackState::TENTATIVE ||
                 signals_[i].state == TrackState::ACTIVE) &&
                signals_[i].freq_hz >= cw_filter_lo_ &&
                signals_[i].freq_hz <= cw_filter_hi_) {
                visible_indices.push_back(i);
            }
        }

        // Clamp selection
        if (selected_signal_ >= static_cast<int>(visible_indices.size()))
            selected_signal_ = std::max(0, static_cast<int>(visible_indices.size()) - 1);

        // === Signal table ===
        std::vector<Element> rows;

        // Header
        rows.push_back(
            hbox({
                text(focus_panel_ == 0 ? " " : "  ") | size(WIDTH, EQUAL, 3),
                text(" FREQ (kHz) ") | bold | color(Color::Cyan),
                text(" SNR ") | bold | color(Color::Green),
                text(" WPM") | bold | color(Color::Yellow),
                text(" CONF ") | bold | color(Color::Magenta),
                text(" STATUS    ") | bold,
            })
        );
        rows.push_back(separator());

        for (int vi = 0; vi < static_cast<int>(visible_indices.size()); vi++) {
            auto& s = signals_[visible_indices[vi]];
            Color state_color = (s.state == TrackState::ACTIVE) ? Color::Green : Color::Yellow;

            int bar_len = std::clamp(static_cast<int>(s.confidence * 10), 0, 10);
            std::string bar(bar_len, '#');
            std::string bar_empty(10 - bar_len, '-');

            char freq_buf[16], snr_buf[8], wpm_buf[8], conf_buf[8];
            snprintf(freq_buf, sizeof(freq_buf), "%10.3f", s.freq_hz / 1000.0);
            snprintf(snr_buf, sizeof(snr_buf), "%3.0f", s.snr_db);
            snprintf(wpm_buf, sizeof(wpm_buf), "%3.0f", s.wpm);
            snprintf(conf_buf, sizeof(conf_buf), "%4.2f", s.confidence);

            bool is_selected = (vi == selected_signal_ && focus_panel_ == 0);
            bool is_playing = (vi == playing_signal_);

            std::string cursor = is_selected ? ">" : " ";
            std::string speaker = is_playing ? "*" : " ";

            auto row = hbox({
                text(cursor + speaker) | (is_selected ? bold : nothing),
                text(std::string(" ") + freq_buf) | color(Color::White),
                text(std::string(snr_buf) + "dB") | color(Color::Green),
                text(std::string(" ") + wpm_buf) | color(Color::Yellow),
                text(std::string(" ") + conf_buf) | color(Color::Magenta),
                text(" "),
                text(bar) | color(state_color),
                text(bar_empty) | dim,
                text(" " + s.status_str()) | color(state_color) | bold,
            });

            if (is_selected) {
                row = row | bgcolor(Color::GrayDark) | focus;
            }
            rows.push_back(row);
        }

        if (visible_indices.empty()) {
            rows.push_back(text("  No CW signals in band") | dim | center);
        }

        // Band name + signal count in panel title
        std::string band_label;
        if (selected_band_ >= 0 && selected_band_ < static_cast<int>(bands_.size())) {
            auto& b = bands_[selected_band_];
            char buf[80];
            snprintf(buf, sizeof(buf), " %s CW: %.0f-%.0f kHz [%d signals] ",
                     b.name.c_str(), b.cw_lo_hz/1000, b.cw_hi_hz/1000,
                     static_cast<int>(visible_indices.size()));
            band_label = buf;
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), " CW Signals [%d] ",
                     static_cast<int>(visible_indices.size()));
            band_label = buf;
        }

        Element signal_panel;
        if (scan_mode_) {
            // === Scan mode: manual frequency dial ===
            char scan_buf[32];
            snprintf(scan_buf, sizeof(scan_buf), "%.3f kHz", scan_freq_ / 1000.0);
            char step_buf[16];
            if (scan_step_ >= 1000) snprintf(step_buf, sizeof(step_buf), "%.0f kHz", scan_step_ / 1000.0);
            else snprintf(step_buf, sizeof(step_buf), "%.0f Hz", scan_step_);

            // Show scan cursor position on a freq scale
            double band_lo = cw_filter_lo_;
            double band_hi = cw_filter_hi_;
            double band_span = band_hi - band_lo;
            int dial_width = 50;
            int cursor_pos = (band_span > 0)
                ? std::clamp(static_cast<int>((scan_freq_ - band_lo) / band_span * dial_width), 0, dial_width - 1)
                : dial_width / 2;

            std::string dial_line(dial_width, '-');
            if (cursor_pos >= 0 && cursor_pos < dial_width) dial_line[cursor_pos] = '|';

            auto scan_rows = vbox({
                text("") | size(HEIGHT, EQUAL, 1),
                text("  Frequency Dial") | bold | color(Color::Cyan) | center,
                text("") | size(HEIGHT, EQUAL, 1),
                text("  " + std::string(scan_buf)) | bold | color(Color::White) | center,
                text("") | size(HEIGHT, EQUAL, 1),
                text("  [" + dial_line + "]") | color(Color::Green) | center,
                hbox({
                    text("  " + format_freq(band_lo) + " kHz") | dim,
                    filler(),
                    text(format_freq(band_hi) + " kHz  ") | dim,
                }),
                text("") | size(HEIGHT, EQUAL, 1),
                text("  Step: " + std::string(step_buf)) | dim | center,
                text("  +/- to change step size") | dim | center,
                text("") | size(HEIGHT, EQUAL, 1),
                hbox({
                    text("  Labels: "),
                    text("CW:" + std::to_string(scan_labels_cw_)) | color(Color::Green),
                    text(" "),
                    text("N:" + std::to_string(scan_labels_noise_)) | color(Color::Red),
                    text(" "),
                    text("?:" + std::to_string(scan_labels_unknown_)) | color(Color::Yellow),
                }) | center,
            });

            signal_panel = window(
                text(" SCAN MODE ") | bold | color(Color::Yellow),
                scan_rows | flex
            ) | flex;
        } else {
            // === Normal signal list with scrolling ===
            auto signal_panel_inner = window(
                text(band_label) | bold | color(Color::Cyan),
                vbox(std::move(rows)) | vscroll_indicator | yframe | flex
            ) | flex;
            signal_panel = signal_panel_inner;
        }

        // === Mini spectrum bar ===
        Element spectrum_el;
        if (!spectrum_db_.empty() && spectrum_bins_ > 0) {
            int ncols = 60;
            std::vector<float> sorted = spectrum_db_;
            std::sort(sorted.begin(), sorted.end());
            float noise_floor = sorted[sorted.size() / 2];
            float peak = *std::max_element(spectrum_db_.begin(), spectrum_db_.end());

            const char levels[] = " .:-=+*#%@";
            std::string bar_line;
            for (int c = 0; c < ncols; c++) {
                int bin_start = c * spectrum_bins_ / ncols;
                int bin_end = (c + 1) * spectrum_bins_ / ncols;
                float max_val = -200.0f;
                for (int b = bin_start; b < bin_end && b < spectrum_bins_; b++) {
                    if (spectrum_db_[b] > max_val) max_val = spectrum_db_[b];
                }
                float frac = std::clamp((max_val - noise_floor) / (peak - noise_floor + 1e-6f), 0.0f, 1.0f);
                int idx = std::clamp(static_cast<int>(frac * 9), 0, 9);
                bar_line += levels[idx];
            }

            double lo_freq = spectrum_center_ - (spectrum_bins_ / 2.0) * spectrum_bin_hz_;
            double hi_freq = spectrum_center_ + (spectrum_bins_ / 2.0) * spectrum_bin_hz_;

            spectrum_el = window(
                text(" Band Activity ") | bold | color(Color::Yellow),
                vbox({
                    text("  " + bar_line) | color(Color::Green),
                    hbox({
                        text("  " + format_freq(lo_freq) + " kHz") | dim,
                        filler(),
                        text(format_freq(spectrum_center_) + " kHz") | dim,
                        filler(),
                        text(format_freq(hi_freq) + " kHz  ") | dim,
                    }),
                })
            );
        } else {
            spectrum_el = window(
                text(" Band Activity ") | bold,
                text("  Waiting for data...") | dim
            );
        }

        // === Right panel ===
        // Band selector
        std::vector<Element> band_rows;
        for (int bi = 0; bi < static_cast<int>(bands_.size()); bi++) {
            bool is_sel = (bi == selected_band_);
            bool is_focused = (focus_panel_ == 1);
            std::string marker = (is_sel ? "> " : "  ");

            char label[32];
            snprintf(label, sizeof(label), "%s %7.0fk",
                     bands_[bi].name.c_str(), bands_[bi].center_hz / 1000.0);

            auto row = text(marker + label);
            if (is_sel && is_focused) {
                row = row | bold | bgcolor(Color::Blue) | color(Color::White);
            } else if (is_sel) {
                row = row | bold | color(Color::Green);
            } else {
                row = row | dim;
            }
            band_rows.push_back(row);
        }

        // Controls help
        std::string focus_names[] = {"Signals", "Bands", "Audio"};
        std::string focus_label = focus_names[focus_panel_];

        std::vector<Element> right_elements;

        right_elements.push_back(
            window(
                text(" Ham Bands ") | bold | color(Color::Yellow),
                vbox(std::move(band_rows))
            )
        );

        if (validation_mode_) {
            // Audio devices
            std::vector<Element> dev_rows;
            for (int di = 0; di < static_cast<int>(audio_devices_.size()); di++) {
                bool is_sel = (di == selected_device_);
                bool is_focused = (focus_panel_ == 2);
                std::string marker = is_sel ? "> " : "  ";
                auto row = text(marker + audio_devices_[di].nick);
                if (is_sel && is_focused) {
                    row = row | bold | bgcolor(Color::Blue) | color(Color::White);
                } else if (is_sel) {
                    row = row | bold | color(Color::Cyan);
                } else {
                    row = row | dim;
                }
                dev_rows.push_back(row);
            }

            right_elements.push_back(
                window(
                    text(" Audio ") | bold | color(Color::Magenta),
                    vbox(std::move(dev_rows)) | vscroll_indicator | yframe | size(HEIGHT, LESS_THAN, 6)
                )
            );
        }

        // Key help
        right_elements.push_back(
            window(
                text(" Keys ") | bold,
                vbox({
                    text(" Tab  [" + focus_label + "]") | color(Color::Cyan),
                    text(" \\/ Navigate") | bold,
                    text(" Enter Select") | bold,
                    validation_mode_
                        ? vbox({
                            text(" F    Scan mode") | bold | color(scan_mode_ ? Color::Yellow : Color::White),
                            text(scan_mode_ ? " </>  Tune freq" : " Space Listen") | bold,
                            text(" Y    CW") | bold | color(Color::Green),
                            text(" N    Noise") | bold | color(Color::Red),
                            text(" U    Unknown") | bold | color(Color::Yellow),
                            text(" T    Test tone") | dim,
                            hbox({
                                text(" CW:") | color(Color::Green),
                                text(std::to_string(labels_positive_ + scan_labels_cw_)),
                                text(" N:") | color(Color::Red),
                                text(std::to_string(labels_negative_ + scan_labels_noise_)),
                                text(" ?:") | color(Color::Yellow),
                                text(std::to_string(labels_unknown_ + scan_labels_unknown_)),
                            }),
                        })
                        : text("") | nothing,
                    text(" q    Quit") | dim,
                })
            )
        );

        Element right_panel = vbox(std::move(right_elements)) | size(WIDTH, EQUAL, 24);

        // === Status bar ===
        std::string band_str;
        if (selected_band_ >= 0 && selected_band_ < static_cast<int>(bands_.size()))
            band_str = bands_[selected_band_].name;

        auto status_bar = hbox({
            text(" SDR: ") | bold,
            text(sdr_device_) | color(sdr_connected_ ? Color::Green : Color::Red),
            separator(),
            text(" ") | size(WIDTH, EQUAL, 1),
            text(format_freq(sdr_freq_) + " kHz") | color(Color::Cyan),
            (!band_str.empty() ? text(" " + band_str) | bold | color(Color::Yellow) : text("")),
            separator(),
            text(" Ch:" + std::to_string(channel_count_)) | dim,
            separator(),
            text(" Up:" + format_uptime(uptime_s_)) | dim,
            filler(),
            text(status_msg_ + " ") | dim,
        }) | border;

        // === Layout ===
        return vbox({
            text(" cwiden CW Signal Identifier ") | bold | center | color(Color::Cyan),
            hbox({
                signal_panel | flex,
                right_panel,
            }) | flex,
            spectrum_el,
            status_bar,
        });
    });

    // Keyboard handler
    auto with_keys = CatchEvent(renderer, [&](Event event) {
        std::lock_guard<std::mutex> lock(mutex_);

        int vis_count = 0;
        for (auto& s : signals_) {
            if ((s.state == TrackState::TENTATIVE || s.state == TrackState::ACTIVE) &&
                s.freq_hz >= cw_filter_lo_ && s.freq_hz <= cw_filter_hi_)
                vis_count++;
        }
        int band_count = static_cast<int>(bands_.size());
        int dev_count = static_cast<int>(audio_devices_.size());

        // Tab: cycle focus
        if (event == Event::Tab) {
            int max_panels = validation_mode_ ? 3 : 2;
            focus_panel_ = (focus_panel_ + 1) % max_panels;
            return true;
        }

        // Quit
        if (event == Event::Character('q') || event == Event::Character('Q')) {
            screen_.Exit();
            return true;
        }

        // Toggle scan mode
        if (validation_mode_ &&
            (event == Event::Character('f') || event == Event::Character('F'))) {
            scan_mode_ = !scan_mode_;
            if (scan_mode_) {
                // Initialize scan freq to band center
                if (selected_band_ >= 0 && selected_band_ < static_cast<int>(bands_.size())) {
                    scan_freq_ = bands_[selected_band_].cw_lo_hz;
                }
                focus_panel_ = 0;  // focus on scan
                status_msg_ = "SCAN: \u2190/\u2192 to tune, Space to play, Y/N to label";
            } else {
                status_msg_ = "Scan mode off";
            }
            return true;
        }

        // Test tone
        if (validation_mode_ &&
            (event == Event::Character('t') || event == Event::Character('T'))) {
            if (test_tone_cb_) {
                test_tone_cb_();
                status_msg_ = "Playing test tone...";
            }
            return true;
        }

        // Navigation
        if (event == Event::ArrowUp || event == Event::Character('k')) {
            if (focus_panel_ == 0 && !scan_mode_ && selected_signal_ > 0) selected_signal_--;
            else if (focus_panel_ == 1 && selected_band_ > 0) selected_band_--;
            else if (focus_panel_ == 2 && selected_device_ > 0) selected_device_--;
            return true;
        }
        if (event == Event::ArrowDown || event == Event::Character('j')) {
            if (focus_panel_ == 0 && !scan_mode_ && selected_signal_ < vis_count - 1) selected_signal_++;
            else if (focus_panel_ == 1 && selected_band_ < band_count - 1) selected_band_++;
            else if (focus_panel_ == 2 && selected_device_ < dev_count - 1) selected_device_++;
            return true;
        }

        // Scan mode: left/right tuning
        if (scan_mode_ && focus_panel_ == 0) {
            if (event == Event::ArrowLeft || event == Event::Character('h')) {
                scan_freq_ -= scan_step_;
                if (scan_freq_ < cw_filter_lo_) scan_freq_ = cw_filter_lo_;
                char buf[64];
                snprintf(buf, sizeof(buf), "Tune: %.3f kHz", scan_freq_ / 1000.0);
                status_msg_ = buf;
                return true;
            }
            if (event == Event::ArrowRight || event == Event::Character('l')) {
                scan_freq_ += scan_step_;
                if (scan_freq_ > cw_filter_hi_) scan_freq_ = cw_filter_hi_;
                char buf[64];
                snprintf(buf, sizeof(buf), "Tune: %.3f kHz", scan_freq_ / 1000.0);
                status_msg_ = buf;
                return true;
            }
            // +/- to change step size
            if (event == Event::Character('+') || event == Event::Character('=')) {
                if (scan_step_ < 10000) scan_step_ *= 2;
                return true;
            }
            if (event == Event::Character('-') || event == Event::Character('_')) {
                if (scan_step_ > 50) scan_step_ /= 2;
                return true;
            }
            // Space: play audio at scan frequency
            if (event == Event::Character(' ')) {
                if (scan_play_cb_) {
                    scan_play_cb_(scan_freq_);
                    char buf[64];
                    snprintf(buf, sizeof(buf), "Playing: %.3f kHz", scan_freq_ / 1000.0);
                    status_msg_ = buf;
                }
                return true;
            }
            // Y/N: label scan frequency
            if (event == Event::Character('y') || event == Event::Character('Y')) {
                if (validation_cb_) validation_cb_(scan_freq_, 1);
                scan_labels_cw_++;
                char buf[64];
                snprintf(buf, sizeof(buf), "CW: %.3f kHz", scan_freq_ / 1000.0);
                status_msg_ = buf;
                // Auto-advance to next frequency
                scan_freq_ += scan_step_;
                if (scan_freq_ > cw_filter_hi_) scan_freq_ = cw_filter_hi_;
                return true;
            }
            if (event == Event::Character('n') || event == Event::Character('N')) {
                if (validation_cb_) validation_cb_(scan_freq_, 0);
                scan_labels_noise_++;
                char buf[64];
                snprintf(buf, sizeof(buf), "NOISE: %.3f kHz", scan_freq_ / 1000.0);
                status_msg_ = buf;
                // Auto-advance to next frequency
                scan_freq_ += scan_step_;
                if (scan_freq_ > cw_filter_hi_) scan_freq_ = cw_filter_hi_;
                return true;
            }
            if (event == Event::Character('u') || event == Event::Character('U')) {
                if (validation_cb_) validation_cb_(scan_freq_, 2);
                scan_labels_unknown_++;
                char buf[64];
                snprintf(buf, sizeof(buf), "UNKNOWN: %.3f kHz", scan_freq_ / 1000.0);
                status_msg_ = buf;
                // Auto-advance to next frequency
                scan_freq_ += scan_step_;
                if (scan_freq_ > cw_filter_hi_) scan_freq_ = cw_filter_hi_;
                return true;
            }
        }

        // Enter/Space: select
        if (event == Event::Return) {
            if (focus_panel_ == 1) {
                // Switch band
                if (selected_band_ >= 0 && selected_band_ < band_count) {
                    auto& b = bands_[selected_band_];
                    cw_filter_lo_ = b.cw_lo_hz;
                    cw_filter_hi_ = b.cw_hi_hz;
                    sdr_freq_ = b.center_hz;  // update display directly
                    if (band_change_cb_) band_change_cb_(selected_band_);
                    status_msg_ = b.name + " band selected";
                }
                return true;
            }
            if (focus_panel_ == 2 && validation_mode_) {
                // Switch audio device
                if (selected_device_ < dev_count) {
                    auto& dev = audio_devices_[selected_device_];
                    if (device_switch_cb_) {
                        bool ok = device_switch_cb_(dev.name);
                        status_msg_ = ok ? ("Audio: " + dev.nick) : ("FAILED: " + dev.nick);
                    }
                }
                return true;
            }
        }

        // Signal-panel actions
        if (focus_panel_ == 0) {
            if (event == Event::Character(' ') && validation_mode_) {
                // Toggle audio
                if (playing_signal_ == selected_signal_) {
                    playing_signal_ = -1;
                    if (audio_select_cb_) audio_select_cb_(-1, 0.0);
                    status_msg_ = "Audio stopped";
                } else {
                    playing_signal_ = selected_signal_;
                    int vi = 0;
                    for (auto& s : signals_) {
                        if ((s.state == TrackState::TENTATIVE || s.state == TrackState::ACTIVE) &&
                            s.freq_hz >= cw_filter_lo_ && s.freq_hz <= cw_filter_hi_) {
                            if (vi == selected_signal_) {
                                if (audio_select_cb_) audio_select_cb_(vi, s.freq_hz);
                                char buf[64];
                                snprintf(buf, sizeof(buf), "Listening: %.3f kHz", s.freq_hz / 1000.0);
                                status_msg_ = buf;
                                break;
                            }
                            vi++;
                        }
                    }
                }
                return true;
            }
            if (event == Event::Character('y') || event == Event::Character('Y')) {
                int vi = 0;
                for (auto& s : signals_) {
                    if ((s.state == TrackState::TENTATIVE || s.state == TrackState::ACTIVE) &&
                        s.freq_hz >= cw_filter_lo_ && s.freq_hz <= cw_filter_hi_) {
                        if (vi == selected_signal_) {
                            if (validation_cb_) validation_cb_(s.freq_hz, 1);
                            labels_positive_++;
                            char buf[64];
                            snprintf(buf, sizeof(buf), "CW: %.3f kHz", s.freq_hz / 1000.0);
                            status_msg_ = buf;
                            break;
                        }
                        vi++;
                    }
                }
                return true;
            }
            if (event == Event::Character('n') || event == Event::Character('N')) {
                int vi = 0;
                for (auto& s : signals_) {
                    if ((s.state == TrackState::TENTATIVE || s.state == TrackState::ACTIVE) &&
                        s.freq_hz >= cw_filter_lo_ && s.freq_hz <= cw_filter_hi_) {
                        if (vi == selected_signal_) {
                            if (validation_cb_) validation_cb_(s.freq_hz, 0);
                            labels_negative_++;
                            char buf[64];
                            snprintf(buf, sizeof(buf), "NOISE: %.3f kHz", s.freq_hz / 1000.0);
                            status_msg_ = buf;
                            break;
                        }
                        vi++;
                    }
                }
                return true;
            }
            if (event == Event::Character('u') || event == Event::Character('U')) {
                int vi = 0;
                for (auto& s : signals_) {
                    if ((s.state == TrackState::TENTATIVE || s.state == TrackState::ACTIVE) &&
                        s.freq_hz >= cw_filter_lo_ && s.freq_hz <= cw_filter_hi_) {
                        if (vi == selected_signal_) {
                            if (validation_cb_) validation_cb_(s.freq_hz, 2);
                            labels_unknown_++;
                            char buf[64];
                            snprintf(buf, sizeof(buf), "UNKNOWN: %.3f kHz", s.freq_hz / 1000.0);
                            status_msg_ = buf;
                            break;
                        }
                        vi++;
                    }
                }
                return true;
            }
        }

        return false;
    });

    // Periodic UI refresh
    std::atomic<bool> ui_running{true};
    std::thread refresh_thread([&] {
        while (ui_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            screen_.PostEvent(Event::Custom);
        }
    });

    screen_.Loop(with_keys);

    ui_running = false;
    if (refresh_thread.joinable()) refresh_thread.join();
}
