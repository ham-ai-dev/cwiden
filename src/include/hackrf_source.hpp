#pragma once
/**
 * hackrf_source.hpp — HackRF SDR device driver.
 * Ported from cwneural.
 */

#include "sdr_source.hpp"
#include <atomic>
#include <libhackrf/hackrf.h>

class HackRFSource : public SDRSource {
public:
    explicit HackRFSource(RingBufPtr ring_buf);
    ~HackRFSource() override;

    bool init(double center_freq_hz, uint32_t sample_rate,
              uint32_t gain1, uint32_t gain2) override;
    bool start() override;
    void stop() override;
    void set_frequency(double freq_hz) override;
    void set_gain1(uint32_t gain_db) override;  // LNA
    void set_gain2(uint32_t gain_db) override;  // VGA
    std::string device_name() const override { return "HackRF One"; }

private:
    static int rx_callback(hackrf_transfer* transfer);

    RingBufPtr ring_buf_;
    hackrf_device* device_ = nullptr;
    double center_freq_ = 0.0;
    uint32_t sample_rate_ = 0;
    uint32_t lna_gain_ = 0;
    uint32_t vga_gain_ = 0;
    std::atomic<bool> running_{false};
};
