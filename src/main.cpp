/**
 * cwiden — CW Signal Identifier for SDR IQ Streams
 *
 * Monitors raw IQ from a HackRF (or IQ file), channelizes the band,
 * and identifies frequencies actively transmitting CW signals using
 * a 3-stage DSP cascade.
 *
 * Modes:
 *   Headless: stdout text/JSON output
 *   TUI:      interactive terminal display
 *   Validate: TUI with audio playback + Y/N labeling for threshold tuning
 */

#include <iostream>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <csignal>
#include <chrono>
#include <fstream>
#include <cstring>
#include <vector>
#include <complex>
#include <mutex>

#include "ring_buffer.hpp"
#include "hackrf_source.hpp"
#include "channelizer.hpp"
#include "cw_classifier.hpp"
#include "signal_tracker.hpp"
#include "audio_output.hpp"
#include "feature_log.hpp"
#include "band_plan.hpp"
#include "tui.hpp"

std::atomic<bool> g_running{true};
void signal_handler(int) { g_running = false; }

// Per-channel IQ cache for audio playback
static std::mutex g_channel_mutex;
struct CachedChannel {
    double freq_hz;
    float bandwidth_hz;
    std::vector<std::complex<float>> iq;
};
static std::vector<CachedChannel> g_cached_channels;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n\n"
              << "SDR options:\n"
              << "  --center-freq <Hz>     SDR center frequency (default: 6933000)\n"
              << "  --sample-rate <Hz>     SDR sample rate (default: 200000)\n"
              << "  --lna-gain <dB>        HackRF LNA gain (default: 16)\n"
              << "  --vga-gain <dB>        HackRF VGA gain (default: 20)\n\n"
              << "Display options:\n"
              << "  --tui                  Launch TUI (default: headless)\n"
              << "  --validate             TUI + audio playback + labeling mode\n"
              << "  --format <text|json>   Headless output format (default: text)\n"
              << "  --interval <seconds>   Output update interval (default: 2.0)\n\n"
              << "Classifier options:\n"
              << "  --min-snr <dB>         Min SNR for channel detection (default: 3)\n"
              << "  --channel-bw <Hz>      Per-channel bandwidth (default: 200)\n"
              << "  --fft-size <N>         FFT size for PSD (default: 8192)\n\n"
              << "Tuning options:\n"
              << "  --tune <data.csv>      Compute optimal thresholds from labeled data\n"
              << "  --log <file.csv>       Log features to CSV (auto in validate mode)\n\n"
              << "Audio options:\n"
              << "  --audio-device <name>  PipeWire sink name\n"
              << "  --list-audio           List available audio output devices\n\n"
              << "Band options:\n"
              << "  --band <name>          Start on a band (160m/80m/40m/30m/20m/17m/15m/12m/10m)\n\n"
              << "Debug options:\n"
              << "  --iq <file.raw>        Replay from raw IQ file (int8 pairs)\n"
              << "  --verbose              Enable debug logging\n"
              << "  --help                 Show this help\n";
}

int main(int argc, char** argv) {
    // Defaults — optimized for CW sub-band scanning
    bool tui_mode = false;
    bool validate_mode = false;
    bool verbose = false;
    double center_freq = 6933000.0;
    uint32_t sample_rate = 200000;
    uint32_t lna_gain = 16;
    uint32_t vga_gain = 20;
    float min_snr = 3.0f;
    float channel_bw = 200.0f;
    int fft_size = 8192;
    float interval = 2.0f;
    std::string format = "text";
    std::string iq_file;
    std::string log_file;
    std::string tune_file;
    std::string audio_device;
    std::string start_band = "40m";
    bool list_audio = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tui") { tui_mode = true; }
        else if (arg == "--validate") { validate_mode = true; tui_mode = true; }
        else if (arg == "--verbose") { verbose = true; }
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else if (arg == "--center-freq" && i+1 < argc) { center_freq = std::stod(argv[++i]); }
        else if (arg == "--sample-rate" && i+1 < argc) { sample_rate = std::stoul(argv[++i]); }
        else if (arg == "--lna-gain" && i+1 < argc) { lna_gain = std::stoul(argv[++i]); }
        else if (arg == "--vga-gain" && i+1 < argc) { vga_gain = std::stoul(argv[++i]); }
        else if (arg == "--min-snr" && i+1 < argc) { min_snr = std::stof(argv[++i]); }
        else if (arg == "--channel-bw" && i+1 < argc) { channel_bw = std::stof(argv[++i]); }
        else if (arg == "--fft-size" && i+1 < argc) { fft_size = std::stoi(argv[++i]); }
        else if (arg == "--interval" && i+1 < argc) { interval = std::stof(argv[++i]); }
        else if (arg == "--format" && i+1 < argc) { format = argv[++i]; }
        else if (arg == "--iq" && i+1 < argc) { iq_file = argv[++i]; }
        else if (arg == "--log" && i+1 < argc) { log_file = argv[++i]; }
        else if (arg == "--tune" && i+1 < argc) { tune_file = argv[++i]; }
        else if (arg == "--audio-device" && i+1 < argc) { audio_device = argv[++i]; }
        else if (arg == "--list-audio") { list_audio = true; }
        else if (arg == "--band" && i+1 < argc) { start_band = argv[++i]; }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // =====================================================================
    // List audio devices
    // =====================================================================
    if (list_audio) {
        std::cerr << "Available audio output devices:" << std::endl;
        std::cerr << "(Use the sink name with --audio-device)" << std::endl;
        std::cerr << std::endl;
        int rc = system("bash scripts/list_audio.sh 2>/dev/null");
        if (rc != 0) {
            system("wpctl status 2>/dev/null | sed -n '/Sinks:/,/Sources:/p'");
        }
        return 0;
    }

    // =====================================================================
    // Tune mode: compute thresholds from labeled data, then exit
    // =====================================================================
    if (!tune_file.empty()) {
        std::cerr << "cwiden: Loading training data from " << tune_file << std::endl;
        auto data = FeatureLog::load(tune_file);
        if (data.empty()) {
            std::cerr << "ERROR: No data in " << tune_file << std::endl;
            return 1;
        }
        auto thresholds = compute_thresholds(data);
        thresholds.print();

        std::string out_path = "thresholds.json";
        if (thresholds.save(out_path)) {
            std::cerr << "Saved to " << out_path << std::endl;
        }
        return 0;
    }

    bool replay_mode = !iq_file.empty();

    // Feature logging
    FeatureLog feature_log;
    bool logging = false;
    if (validate_mode && log_file.empty()) {
        log_file = "training_data.csv";
    }
    if (!log_file.empty()) {
        if (feature_log.open(log_file)) {
            logging = true;
            std::cerr << "cwiden: Logging features to " << log_file << std::endl;
        }
    }

    // Audio output (validate mode only)
    AudioOutput audio;
    if (validate_mode) {
        if (!audio.init(8000, 700.0f, audio_device)) {
            std::cerr << "WARNING: Audio init failed, continuing without audio" << std::endl;
        } else if (!audio_device.empty()) {
            std::cerr << "cwiden: Audio output: " << audio_device << std::endl;
        }
    }

    // =====================================================================
    // IQ File Replay Mode
    // =====================================================================
    if (replay_mode) {
        std::cerr << "cwiden: Replay mode — " << iq_file << std::endl;

        std::ifstream f(iq_file, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            std::cerr << "ERROR: Cannot open " << iq_file << std::endl;
            return 1;
        }
        size_t file_size = f.tellg();
        f.seekg(0);
        std::vector<int8_t> raw(file_size);
        f.read(reinterpret_cast<char*>(raw.data()), file_size);

        size_t num_samples = file_size / 2;
        std::vector<std::complex<float>> iq(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            iq[i] = std::complex<float>(raw[2*i] / 128.0f, raw[2*i+1] / 128.0f);
        }

        std::cerr << "  Loaded " << num_samples << " samples ("
                  << num_samples / (float)sample_rate << "s)" << std::endl;

        Channelizer channelizer(sample_rate, center_freq, channel_bw, min_snr, fft_size);
        CWClassifier classifier;
        SignalTracker tracker;

        int chunk_size = sample_rate * 2;
        double elapsed = 0.0;

        for (size_t offset = 0; offset + chunk_size <= num_samples && g_running; offset += chunk_size) {
            auto channels = channelizer.process(iq.data() + offset, chunk_size);

            std::vector<std::tuple<double, float, float, float>> detections;
            for (auto& ch : channels) {
                auto result = classifier.classify(ch.iq.data(),
                                                   static_cast<int>(ch.iq.size()),
                                                   ch.bandwidth_hz);
                if (verbose) {
                    std::cerr << "  " << ch.center_freq_hz / 1000.0 << " kHz: "
                              << signal_type_str(result.signal_class)
                              << " conf=" << result.cw_confidence
                              << " — " << result.verdict << std::endl;
                }

                if (logging) {
                    FeatureRow row;
                    row.timestamp = elapsed;
                    row.freq_hz = ch.center_freq_hz;
                    row.snr_db = ch.snr_db;
                    row.effective_bw = result.features.effective_bw;
                    row.shape_factor = result.features.shape_factor;
                    row.headroom_db = result.features.headroom_db;
                    row.bimodality_coeff = result.features.bimodality_coeff;
                    row.on_off_ratio = result.features.on_off_ratio;
                    row.rhythm_score = result.features.rhythm_score;
                    row.wpm_estimate = result.wpm_estimate;
                    row.confidence = result.cw_confidence;
                    row.classified_cw = result.is_cw;
                    row.stage_reached = result.stage_reached;
                    feature_log.log(row);
                }

                if (result.is_cw || result.cw_confidence > 0.3f) {
                    detections.emplace_back(ch.center_freq_hz, ch.snr_db,
                                             result.wpm_estimate, result.cw_confidence);
                }
            }

            elapsed += 2.0;
            tracker.update(detections, elapsed);

            if (format == "json") {
                auto json = tracker.format_json(elapsed);
                if (!json.empty()) std::cout << json << std::endl;
            } else {
                auto text = tracker.format_text(elapsed);
                if (!text.empty()) std::cout << text << std::flush;
            }
        }

        std::cerr << "\ncwiden: Replay complete." << std::endl;
        return 0;
    }

    // =====================================================================
    // Live HackRF Mode
    // =====================================================================
    auto ring_buf = std::make_shared<RingBuffer<std::complex<float>>>(1 << 20);
    HackRFSource hackrf(ring_buf);

    if (!hackrf.init(center_freq, sample_rate, lna_gain, vga_gain)) {
        std::cerr << "ERROR: Failed to initialize HackRF" << std::endl;
        return 1;
    }

    Channelizer channelizer(sample_rate, center_freq, channel_bw, min_snr, fft_size);
    CWClassifier classifier;
    SignalTracker tracker;

    // TUI setup
    if (tui_mode) {
        // Set up band presets
        auto presets = get_band_presets();
        std::vector<Tui::BandInfo> tui_bands;
        int start_idx = 2; // default to 40m
        for (int i = 0; i < static_cast<int>(presets.size()); i++) {
            tui_bands.push_back({
                presets[i].name,
                presets[i].center_hz,
                presets[i].cw_lo_hz,
                presets[i].cw_hi_hz
            });
            if (presets[i].name == start_band) start_idx = i;
        }
        Tui::set_bands(tui_bands);

        // Apply initial band
        if (start_idx >= 0 && start_idx < static_cast<int>(presets.size())) {
            auto& bp = presets[start_idx];
            center_freq = bp.center_hz;
            sample_rate = bp.sample_rate;
            hackrf.set_frequency(center_freq);
            channelizer.set_center_freq(center_freq);
            Tui::set_current_band(start_idx);
            Tui::set_cw_filter(bp.cw_lo_hz, bp.cw_hi_hz);
        }

        Tui::set_initial_config(center_freq, lna_gain, vga_gain, start_band);
        Tui::update_sdr_info("HackRF", center_freq, sample_rate, true);

        // Band change callback — retunes HackRF
        Tui::set_band_change_callback([&](int band_idx) {
            auto presets = get_band_presets();
            if (band_idx >= 0 && band_idx < static_cast<int>(presets.size())) {
                auto& bp = presets[band_idx];
                hackrf.set_frequency(bp.center_hz);
                channelizer.set_center_freq(bp.center_hz);
                center_freq = bp.center_hz;
                Tui::update_sdr_info("HackRF", center_freq, sample_rate, true);
            }
        });

        if (validate_mode) {
            Tui::set_validation_mode(true);

            auto devs = AudioOutput::list_devices();
            std::vector<Tui::DeviceInfo> tui_devs;
            for (auto& d : devs) {
                tui_devs.push_back({d.name, d.nick});
            }
            Tui::set_audio_devices(tui_devs);

            Tui::set_device_switch_callback([&](const std::string& dev_name) -> bool {
                return audio.set_device(dev_name);
            });

            Tui::set_test_tone_callback([&]() {
                audio.play_test_tone();
            });

            Tui::set_validation_callback([&](double freq_hz, int label) {
                if (logging) {
                    feature_log.update_label(freq_hz, label);
                }
            });

            Tui::set_audio_select_callback([&](int signal_idx, double freq_hz) {
                if (signal_idx < 0) return;
                std::lock_guard<std::mutex> lock(g_channel_mutex);
                for (auto& cc : g_cached_channels) {
                    if (std::abs(cc.freq_hz - freq_hz) < 100.0) {
                        audio.play(cc.iq, cc.bandwidth_hz);
                        return;
                    }
                }
            });
        }

        Tui::set_config_change_callback([&](const std::string& key, const std::string& val) {
            try {
                if (key == "center_freq") {
                    double f = std::stod(val);
                    hackrf.set_frequency(f);
                    channelizer.set_center_freq(f);
                    center_freq = f;
                } else if (key == "lna_gain") {
                    hackrf.set_gain1(std::stoul(val));
                } else if (key == "vga_gain") {
                    hackrf.set_gain2(std::stoul(val));
                }
            } catch (...) {}
        });
    }

    if (!hackrf.start()) {
        std::cerr << "ERROR: Failed to start HackRF RX" << std::endl;
        return 1;
    }

    std::cerr << "cwiden: Live mode — center=" << center_freq / 1e6 << " MHz"
              << " rate=" << sample_rate / 1e6 << " Msps"
              << (validate_mode ? " [VALIDATE]" : "")
              << " (Ctrl+C to stop)" << std::endl;

    auto start_time = std::chrono::steady_clock::now();

    // Processing thread
    std::thread pipeline_thread([&] {
        std::vector<std::complex<float>> iq_batch(8192);
        std::vector<std::complex<float>> accumulator;
        int chunk_size = sample_rate * 2;
        accumulator.reserve(chunk_size + 8192);

        while (g_running) {
            int got = 0;
            for (int i = 0; i < 8192; i++) {
                if (ring_buf->pop(iq_batch[i])) got++;
                else break;
            }

            if (got == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            accumulator.insert(accumulator.end(), iq_batch.begin(), iq_batch.begin() + got);

            if (static_cast<int>(accumulator.size()) >= chunk_size) {
                auto channels = channelizer.process(accumulator.data(), chunk_size);

                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();

                // Cache channel IQ for audio playback in validate mode
                if (validate_mode) {
                    std::lock_guard<std::mutex> lock(g_channel_mutex);
                    g_cached_channels.clear();
                    for (auto& ch : channels) {
                        g_cached_channels.push_back({
                            ch.center_freq_hz,
                            ch.bandwidth_hz,
                            ch.iq
                        });
                    }
                }

                std::vector<std::tuple<double, float, float, float>> detections;
                for (auto& ch : channels) {
                    auto result = classifier.classify(ch.iq.data(),
                                                       static_cast<int>(ch.iq.size()),
                                                       ch.bandwidth_hz);
                    if (verbose) {
                        std::cerr << "  " << ch.center_freq_hz / 1000.0 << " kHz: "
                                  << signal_type_str(result.signal_class)
                                  << " conf=" << result.cw_confidence
                                  << " — " << result.verdict << std::endl;
                    }

                    if (logging) {
                        FeatureRow row;
                        row.timestamp = elapsed;
                        row.freq_hz = ch.center_freq_hz;
                        row.snr_db = ch.snr_db;
                        row.effective_bw = result.features.effective_bw;
                        row.shape_factor = result.features.shape_factor;
                        row.headroom_db = result.features.headroom_db;
                        row.bimodality_coeff = result.features.bimodality_coeff;
                        row.on_off_ratio = result.features.on_off_ratio;
                        row.rhythm_score = result.features.rhythm_score;
                        row.wpm_estimate = result.wpm_estimate;
                        row.confidence = result.cw_confidence;
                        row.classified_cw = result.is_cw;
                        row.stage_reached = result.stage_reached;
                        feature_log.log(row);
                    }

                    if (result.is_cw || result.cw_confidence > 0.3f) {
                        detections.emplace_back(ch.center_freq_hz, ch.snr_db,
                                                 result.wpm_estimate, result.cw_confidence);
                    }
                }

                tracker.update(detections, elapsed);

                if (tui_mode) {
                    Tui::update_signals(tracker.signals());
                    Tui::update_spectrum(channelizer.last_psd_db(),
                                         channelizer.psd_size(),
                                         channelizer.center_freq(),
                                         channelizer.freq_per_bin());
                    Tui::set_channel_count(static_cast<int>(channels.size()));
                    Tui::set_uptime(elapsed);
                } else {
                    if (format == "json") {
                        auto json = tracker.format_json(elapsed);
                        if (!json.empty()) std::cout << json << std::endl;
                    } else {
                        auto text = tracker.format_text(elapsed);
                        if (!text.empty()) std::cout << text << std::flush;
                    }
                }

                if (static_cast<int>(accumulator.size()) > chunk_size) {
                    accumulator.erase(accumulator.begin(),
                                       accumulator.begin() + chunk_size);
                } else {
                    accumulator.clear();
                }
            }
        }
    });

    if (tui_mode) {
        Tui tui;
        tui.run();
        g_running = false;
    } else {
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    pipeline_thread.join();
    if (validate_mode) {
        audio.stop();
    }
    hackrf.stop();
    std::cerr << "\ncwiden: Shutdown complete." << std::endl;
    return 0;
}
