#pragma once
#include <complex>
#include <vector>
#include <atomic>
#include <string>
#include <cstdint>
#include <sys/types.h>

struct AudioDevice {
    std::string name;  // PipeWire sink name (for pw-play --target)
    std::string nick;  // Short display name
    std::string desc;  // Full description
};

class AudioOutput {
public:
    AudioOutput();
    ~AudioOutput();

    /** List available audio output devices. */
    static std::vector<AudioDevice> list_devices();

    /** Initialize. Empty device = system default (aplay). */
    bool init(int audio_rate = 8000, float bfo_freq = 700.0f,
              const std::string& device = "");

    /** Switch to a different output device. */
    bool set_device(const std::string& device);

    /** Play a short test tone (700 Hz, 2 seconds). */
    void play_test_tone();

    /** Play narrowband IQ through BFO. */
    void play(const std::vector<std::complex<float>>& iq, float channel_rate);

    /** Stop playback. */
    void stop();

    bool is_playing() const { return playing_.load(); }
    void set_bfo(float freq) { bfo_freq_ = freq; }
    void set_volume(float vol) { volume_ = vol; }
    const std::string& current_device() const { return device_name_; }

private:
    void play_pcm(const std::vector<float>& pcm);
    void close_stream();

    int audio_rate_ = 8000;
    float bfo_freq_ = 700.0f;
    float volume_ = 0.7f;
    std::atomic<bool> running_{false};
    std::atomic<bool> playing_{false};
    std::string device_name_;
    std::string wav_path_;
    pid_t play_pid_ = -1;
};
