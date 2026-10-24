#pragma once

// LPN_PROFILE=1: once-a-second reports while playing, to find where and why the fps drops.
// Every line carries t=<seconds since start>; F10 writes a [mark] line at the moment it is
// pressed, so a drop seen on screen can be found in the log.
namespace host::profile {

bool on();
double seconds(); // since the host started
void nameThread(const char* name); // shown in the [cpu] lines; 15 characters at most
void mark(); // F10
void start(); // the [cpu] sampler: CPU use of each thread, once a second

} // namespace host::profile
