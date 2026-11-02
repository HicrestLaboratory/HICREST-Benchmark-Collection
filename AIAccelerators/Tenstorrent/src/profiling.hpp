#pragma once

#include <string>
#include <chrono>
#include <iostream>

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

namespace benchmark::profiling {

inline void signpost(const std::string& phase, const char* boundary) {
    // Capture wall-clock boundaries on the same host clock as the power sampler.
    // Defer output until the end so logging does not delay the measured loop.
    if (phase == "steady_state") {
        static long long start_ns = 0;
        const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (std::string(boundary) == "start") {
            start_ns = now_ns;
        } else if (std::string(boundary) == "end") {
            std::cout << "HICREST_MEASURED_INTERVAL " << start_ns << " " << now_ns << std::endl;
        }
    }
    // Keep the single-line payload free of Tracy's backtick CSV quote
    // character so it cannot consume the following operation row. The perf
    // report analyzes operations after the last TT_SIGNPOST, so reserve that
    // prefix for the beginning of the measured steady-state interval. Other
    // boundaries remain visible as timestamped TT_PHASE Tracy messages.
    const bool is_analysis_boundary = phase == "steady_state" && std::string(boundary) == "start";
    const std::string prefix = is_analysis_boundary ? "TT_SIGNPOST: " : "TT_PHASE: ";
    const std::string message = prefix + phase + ":" + boundary;
    TracyMessageC(message.c_str(), message.size(), 0xf0f8ff);
}

}  // namespace benchmark::profiling
