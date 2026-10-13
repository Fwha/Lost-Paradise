#include "host/equeue.h"

#include "host/hle.h"
#include "host/kernel.h"

#include <algorithm>
#include <chrono>

namespace host {

namespace {

std::mutex gQueuesMutex;
std::vector<EventQueue*> gQueues;

} // namespace

void EventQueue::add(uint64_t ident, int16_t filter, void* udata) {
    std::lock_guard lock(m_);
    for (auto& e : entries_)
        if (e.ev.ident == ident && e.ev.filter == filter) {
            e.ev.udata = udata;
            return;
        }
    entries_.push_back({{ident, filter, 0x20 /*EV_CLEAR*/, 0, 0, udata}, false});
}

void EventQueue::remove(uint64_t ident, int16_t filter) {
    std::lock_guard lock(m_);
    std::erase_if(entries_, [&](const Entry& e) { return e.ev.ident == ident && e.ev.filter == filter; });
}

void EventQueue::trigger(uint64_t ident, int16_t filter, int64_t data) {
    bool any = false;
    {
        std::lock_guard lock(m_);
        for (auto& e : entries_)
            if (e.ev.ident == ident && e.ev.filter == filter) {
                e.fired = true;
                e.ev.data = data;
                any = true;
            }
    }
    if (any) cv_.notify_all();
}

int EventQueue::wait(KernelEvent* out, int max, int64_t timeoutUs) {
    std::unique_lock lock(m_);
    auto ready = [&] { return std::any_of(entries_.begin(), entries_.end(), [](const Entry& e) { return e.fired; }); };
    if (timeoutUs < 0) cv_.wait(lock, ready);
    else if (!cv_.wait_for(lock, std::chrono::microseconds(timeoutUs), ready)) return 0;
    int n = 0;
    for (auto& e : entries_)
        if (e.fired && n < max) {
            out[n++] = e.ev;
            e.fired = false;
        }
    return n;
}

void triggerEvent(uint64_t ident, int16_t filter, int64_t data) {
    std::lock_guard lock(gQueuesMutex);
    for (EventQueue* q : gQueues) q->trigger(ident, filter, data);
}

namespace {

int32_t sceKernelCreateEqueue(EventQueue** out, const char* name) {
    if (!out) return sceError(ps4errno::kEINVAL);
    *out = new EventQueue(name ? name : "");
    std::lock_guard lock(gQueuesMutex);
    gQueues.push_back(*out);
    return 0;
}

int32_t sceKernelDeleteEqueue(EventQueue* q) {
    if (!q) return sceError(ps4errno::kEBADF);
    {
        std::lock_guard lock(gQueuesMutex);
        std::erase(gQueues, q);
    }
    delete q;
    return 0;
}

int32_t sceKernelWaitEqueue(EventQueue* q, KernelEvent* ev, int32_t num, int32_t* out, const uint32_t* timeoutUs) {
    if (!q) return sceError(ps4errno::kEBADF);
    if (!ev || num < 1) return sceError(ps4errno::kEINVAL);
    const int n = q->wait(ev, num, timeoutUs ? int64_t(*timeoutUs) : -1);
    if (out) *out = n;
    return n == 0 ? sceError(ps4errno::kETIMEDOUT) : 0;
}

} // namespace

void registerEqueueHle() {
    const char* k = "libkernel";
    HLE(k, sceKernelCreateEqueue);
    HLE(k, sceKernelDeleteEqueue);
    HLE(k, sceKernelWaitEqueue);
}

} // namespace host
