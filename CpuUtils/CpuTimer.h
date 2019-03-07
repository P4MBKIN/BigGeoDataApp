#pragma once

#define CPU_TIMER_START \
auto start = std::chrono::steady_clock::now();

#define CPU_TIMER_STOP(X) \
auto end = std::chrono::steady_clock::now(); \
auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
X = (double)elapsed.count() / 1000;
