#pragma once

// LPN_PROFILE=1: once-a-second reports while playing, to find where and why the fps drops.
// Every line carries t=<seconds since start>; F10 writes a [mark] line at the moment it is
// pressed, so a drop seen on screen can be found in the log.
#include <atomic>
#include <cstdint>

namespace host::profile {

// Game file and decompression work, reported once a second as [io] t=... lines.
struct IoCounters {
    std::atomic<uint64_t> opens{0}, reads{0}, bytes{0}, readNs{0}, inflates{0}, inflateNs{0};
};
IoCounters& io();

bool on();
double seconds(); // since the host started
void nameThread(const char* name); // shown in the [cpu] lines; 15 characters at most
void lowerThreadPriority(); // the calling thread yields to the game's (background work)
// Keeps the calling thread (and threads it starts) off the core set aside for the recorder
// (see nameThread); every thread but the recorder starts so, from main.
void avoidRecorderCore();
void mark(); // F10
void start(); // the [cpu] sampler (CPU use of each thread) and the [wait] sampler, once a second

} // namespace host::profile
