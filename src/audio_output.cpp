#include "audio_output.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <unistd.h>
#include <sys/wait.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

AudioOutput::AudioOutput() {}

AudioOutput::~AudioOutput() {
    stop();
}

// =========================================================================
// Device enumeration
// =========================================================================
std::vector<AudioDevice> AudioOutput::list_devices() {
    std::vector<AudioDevice> devices;

    FILE* pipe = popen("pw-cli list-objects 2>/dev/null", "r");
    if (!pipe) return devices;

    char buf[512];
    std::string cur_name, cur_nick, cur_desc;

    while (fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);

        auto dpos = line.find("node.description = \"");
        if (dpos != std::string::npos) {
            dpos += 20;
            auto end = line.find('"', dpos);
            if (end != std::string::npos)
                cur_desc = line.substr(dpos, end - dpos);
        }

        auto npos = line.find("node.name = \"");
        if (npos != std::string::npos) {
            npos += 13;
            auto end = line.find('"', npos);
            if (end != std::string::npos)
                cur_name = line.substr(npos, end - npos);
        }

        auto kpos = line.find("node.nick = \"");
        if (kpos != std::string::npos) {
            kpos += 13;
            auto end = line.find('"', kpos);
            if (end != std::string::npos)
                cur_nick = line.substr(kpos, end - kpos);
        }

        if (line.find("Audio/Sink") != std::string::npos) {
            if (!cur_name.empty()) {
                AudioDevice dev;
                dev.name = cur_name;
                dev.nick = cur_nick.empty() ? cur_name : cur_nick;
                dev.desc = cur_desc.empty() ? cur_nick : cur_desc;
                devices.push_back(dev);
            }
            cur_name.clear();
            cur_nick.clear();
            cur_desc.clear();
        }
    }
    pclose(pipe);
    return devices;
}

// =========================================================================
// WAV writing helper
// =========================================================================
static void write_wav(const std::string& path, const std::vector<float>& pcm, int rate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;

    int n = static_cast<int>(pcm.size());
    // Convert to 16-bit PCM
    std::vector<int16_t> samples(n);
    for (int i = 0; i < n; i++) {
        float v = std::clamp(pcm[i], -1.0f, 1.0f);
        samples[i] = static_cast<int16_t>(v * 32767);
    }

    int data_size = n * 2;
    int file_size = 36 + data_size;

    // RIFF header
    fwrite("RIFF", 1, 4, f);
    uint32_t u32 = file_size;
    fwrite(&u32, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    u32 = 16;    fwrite(&u32, 4, 1, f);  // chunk size
    uint16_t u16 = 1;  fwrite(&u16, 2, 1, f);  // PCM
    u16 = 1;           fwrite(&u16, 2, 1, f);  // mono
    u32 = rate;         fwrite(&u32, 4, 1, f);  // sample rate
    u32 = rate * 2;     fwrite(&u32, 4, 1, f);  // byte rate
    u16 = 2;           fwrite(&u16, 2, 1, f);  // block align
    u16 = 16;          fwrite(&u16, 2, 1, f);  // bits per sample

    // data chunk
    fwrite("data", 1, 4, f);
    u32 = data_size;
    fwrite(&u32, 4, 1, f);
    fwrite(samples.data(), 2, n, f);
    fclose(f);
}

// =========================================================================
// Init / device switching
// =========================================================================
void AudioOutput::close_stream() {
    playing_ = false;
    running_ = false;
    // Kill any aplay child
    if (play_pid_ > 0) {
        kill(play_pid_, SIGTERM);
        waitpid(play_pid_, nullptr, 0);
        play_pid_ = -1;
    }
}

bool AudioOutput::init(int audio_rate, float bfo_freq, const std::string& device) {
    close_stream();
    audio_rate_ = audio_rate;
    bfo_freq_ = bfo_freq;
    device_name_ = device;
    running_ = true;
    return true;  // No stream to open — we fork aplay on demand
}

bool AudioOutput::set_device(const std::string& device) {
    close_stream();
    device_name_ = device;
    running_ = true;
    return true;
}

// =========================================================================
// Play audio via aplay (non-blocking fork)
// =========================================================================
void AudioOutput::play_pcm(const std::vector<float>& pcm) {
    close_stream();  // stop previous playback

    // Write WAV to temp file
    wav_path_ = "/tmp/cwiden_audio.wav";
    write_wav(wav_path_, pcm, audio_rate_);

    // Fork aplay with looping
    play_pid_ = fork();
    if (play_pid_ == 0) {
        // Child: loop aplay
        while (true) {
            if (device_name_.empty()) {
                execlp("aplay", "aplay", "-q", wav_path_.c_str(), nullptr);
            } else {
                // Use pw-play with target for device selection
                execlp("pw-play", "pw-play", "--target",
                       device_name_.c_str(), wav_path_.c_str(), nullptr);
            }
            _exit(1);
        }
    }

    playing_ = true;
}

// =========================================================================
// Test tone
// =========================================================================
void AudioOutput::play_test_tone() {
    int n = audio_rate_ * 2;  // 2 seconds
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++) {
        pcm[i] = 0.5f * std::sin(2.0 * M_PI * 700.0 * i / audio_rate_);
    }
    play_pcm(pcm);
}

// =========================================================================
// IQ playback
// =========================================================================
void AudioOutput::play(const std::vector<std::complex<float>>& iq, float channel_rate) {
    int n_in = static_cast<int>(iq.size());
    if (n_in == 0) return;

    float ratio = static_cast<float>(audio_rate_) / channel_rate;
    int n_out = static_cast<int>(n_in * ratio);

    std::vector<float> pcm(n_out);
    double bfo_phase = 0.0;
    double bfo_inc = 2.0 * M_PI * bfo_freq_ / audio_rate_;

    for (int i = 0; i < n_out; i++) {
        float src_pos = i / ratio;
        int idx0 = static_cast<int>(src_pos);
        float frac = src_pos - idx0;
        int idx1 = std::min(idx0 + 1, n_in - 1);
        idx0 = std::min(idx0, n_in - 1);

        std::complex<float> sample = iq[idx0] * (1.0f - frac) + iq[idx1] * frac;

        float bfo_cos = std::cos(bfo_phase);
        float bfo_sin = std::sin(bfo_phase);
        float audio_sample = sample.real() * bfo_cos - sample.imag() * bfo_sin;

        pcm[i] = audio_sample * volume_;
        bfo_phase += bfo_inc;
        if (bfo_phase > 2.0 * M_PI) bfo_phase -= 2.0 * M_PI;
    }

    // Normalize
    float peak = 0.0f;
    for (auto v : pcm) peak = std::max(peak, std::abs(v));
    if (peak > 0.01f) {
        float scale = 0.8f / peak;
        for (auto& v : pcm) v *= scale;
    }

    play_pcm(pcm);
}

// =========================================================================
// Stop
// =========================================================================
void AudioOutput::stop() {
    close_stream();
}
