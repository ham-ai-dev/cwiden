#pragma once
/**
 * sdr_source.hpp — Abstract SDR device interface.
 * Implementations: HackRFSource (future: RTLSDRSource)
 */

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include "ring_buffer.hpp"

class SDRSource {
public:
    virtual ~SDRSource() = default;

    virtual bool init(double center_freq_hz, uint32_t sample_rate,
                      uint32_t gain1, uint32_t gain2) = 0;
    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual void set_frequency(double freq_hz) = 0;
    virtual void set_gain1(uint32_t gain_db) = 0;
    virtual void set_gain2(uint32_t gain_db) = 0;
    virtual std::string device_name() const = 0;

    using RingBufPtr = std::shared_ptr<RingBuffer<std::complex<float>>>;
};
