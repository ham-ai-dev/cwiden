#pragma once
/**
 * ring_buffer.hpp — Lock-free SPSC ring buffer.
 * Ported from cwneural with minor cleanup.
 */

#include <atomic>
#include <vector>
#include <cstddef>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity), buf_(capacity), head_(0), tail_(0) {}

    bool push(const T& item) {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t next = (h + 1) % capacity_;
        if (next == tail_.load(std::memory_order_acquire)) return false; // full
        buf_[h] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false; // empty
        item = buf_[t];
        tail_.store((t + 1) % capacity_, std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t h = head_.load(std::memory_order_acquire);
        size_t t = tail_.load(std::memory_order_acquire);
        return (h >= t) ? (h - t) : (capacity_ - t + h);
    }

    size_t capacity() const { return capacity_; }
    bool empty() const { return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire); }

private:
    size_t capacity_;
    std::vector<T> buf_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};
