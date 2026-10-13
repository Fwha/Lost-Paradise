// Kernel event queues (kqueue). The game waits on these for display flips and GPU
// end-of-pipe interrupts; VideoOut and GNM trigger the events.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace host {

// Filters as the PS4 numbers them.
constexpr int16_t kFilterVideoOut = -13;
constexpr int16_t kFilterGraphicsCore = -14;

// struct kevent as the guest sees it.
struct KernelEvent {
    uint64_t ident;
    int16_t filter;
    uint16_t flags;
    uint32_t fflags;
    int64_t data;
    void* udata;
};
static_assert(sizeof(KernelEvent) == 32);

class EventQueue {
public:
    explicit EventQueue(std::string name) : name_(std::move(name)) {}

    void add(uint64_t ident, int16_t filter, void* udata);
    void remove(uint64_t ident, int16_t filter);
    // Mark an event as fired. Events are edge-triggered (EV_CLEAR): waiting consumes them.
    void trigger(uint64_t ident, int16_t filter, int64_t data);
    // Returns the number of events written; 0 on timeout. timeoutUs < 0 waits forever.
    int wait(KernelEvent* out, int max, int64_t timeoutUs);

    const std::string& name() const { return name_; }

private:
    struct Entry {
        KernelEvent ev;
        bool fired = false;
    };
    std::string name_;
    std::mutex m_;
    std::condition_variable cv_;
    std::vector<Entry> entries_;
};

// Fire an event on every queue that registered it.
void triggerEvent(uint64_t ident, int16_t filter, int64_t data);

void registerEqueueHle();

} // namespace host
