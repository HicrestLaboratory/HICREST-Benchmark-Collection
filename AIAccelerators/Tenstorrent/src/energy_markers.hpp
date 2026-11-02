#pragma once
#include <chrono>
#include <cstdint>
#include <iostream>

namespace benchmark::energy {
inline std::int64_t unix_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
// Capture boundaries before printing; output is outside the measured interval.
class Interval {
    const char* marker_;
    std::int64_t start_;
    bool active_ = true;
public:
    explicit Interval(const char* marker) : marker_(marker), start_(unix_ns()) {}
    Interval(const Interval&) = delete;
    Interval& operator=(const Interval&) = delete;
    void finish() {
        if (!active_) return;
        const auto end = unix_ns();
        active_ = false;
        std::cout << marker_ << " " << start_ << " " << end << std::endl;
    }
    ~Interval() { finish(); }
};
inline void work(std::uint64_t runs, std::uint64_t per_run, const char* unit) {
    std::cout << "HICREST_MEASURED_WORK " << runs << " " << per_run << " " << unit << std::endl;
}
}  // namespace benchmark::energy
