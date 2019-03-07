#pragma once

#define GPU_TIMER_START \
cudaEvent_t start, stop; \
cudaEventCreate(&start); \
cudaEventCreate(&stop); \
cudaEventRecord(start, 0);

#define GPU_TIMER_STOP(X) \
cudaEventRecord(stop, 0); \
cudaEventSynchronize(stop); \
cudaEventElapsedTime(&X, start, stop);
