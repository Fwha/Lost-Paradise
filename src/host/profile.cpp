#include "host/profile.h"

#include "host/crash.h"
#include "host/hle.h"
#include "host/linker.h"
#include "host/module.h"
#include "host/pm4.h"
#include "host/videoout.h"

#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace host::profile {
namespace {

const auto gStart = std::chrono::steady_clock::now();
std::atomic<int> gMarks{0};

struct Task {
    std::string name;
    uint64_t ticks; // user + system
};

// Every thread of this process with its CPU time so far, from /proc/self/task/<tid>/stat.
std::unordered_map<int, Task> readTasks() {
    std::unordered_map<int, Task> out;
    DIR* d = opendir("/proc/self/task");
    if (!d) return out;
    while (dirent* e = readdir(d)) {
        const int tid = std::atoi(e->d_name);
        if (tid <= 0) continue;
        char path[64], buf[512];
        std::snprintf(path, sizeof path, "/proc/self/task/%d/stat", tid);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        std::fclose(f);
        buf[n] = 0;
        // "tid (name) state ..." : the name may hold spaces and parentheses, so take the last ')'.
        char* open = std::strchr(buf, '(');
        char* close = std::strrchr(buf, ')');
        if (!open || !close || close < open) continue;
        Task t{std::string(open + 1, close), 0};
        // After ')': state is field 3; utime and stime are fields 14 and 15.
        char* p = close + 2;
        unsigned long long utime = 0, stime = 0;
        if (std::sscanf(p, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &utime, &stime) != 2) continue;
        t.ticks = utime + stime;
        out.emplace(tid, std::move(t));
    }
    closedir(d);
    return out;
}

void sampler() {
    nameThread("lpn-prof");
    const double hz = double(sysconf(_SC_CLK_TCK));
    auto before = readTasks();
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto now = readTasks();
        const auto t1 = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(t1 - t0).count();
        // Threads that share a name (pipeline workers, the game's pools) are summed, with a count.
        std::map<std::string, std::pair<double, int>> byName;
        double total = 0;
        for (const auto& [tid, t] : now) {
            auto it = before.find(tid);
            const uint64_t ticks = it != before.end() ? t.ticks - std::min(t.ticks, it->second.ticks) : t.ticks;
            const double pct = 100.0 * double(ticks) / hz / wall;
            total += pct;
            auto& b = byName[t.name];
            b.first += pct, ++b.second;
        }
        std::vector<std::pair<double, std::string>> rows;
        for (const auto& [name, v] : byName)
            if (v.first >= 3) rows.push_back({v.first, v.second > 1 ? name + " x" + std::to_string(v.second) : name});
        std::sort(rows.rbegin(), rows.rend());
        std::string line;
        for (size_t i = 0; i < std::min<size_t>(rows.size(), 12); ++i) {
            char cell[96];
            std::snprintf(cell, sizeof cell, "%s %s %.0f%%", i ? "," : "", rows[i].second.c_str(), rows[i].first);
            line += cell;
        }
        std::fprintf(stderr, "[cpu] t=%.1f total %.0f%%:%s\n", seconds(), total, line.c_str());
        before = std::move(now);
        t0 = t1;
    }
}

// --- [wait]: what the game's threads are blocked in ------------------------------------

std::mutex gSlotsMutex; // held by the sampler while it reads a slot, so its thread's stack stays put
std::vector<WaitSlot*> gSlots;

struct SlotOwner {
    WaitSlot slot;
    SlotOwner() {
        slot.tid = int(syscall(SYS_gettid));
        std::lock_guard lock(gSlotsMutex);
        gSlots.push_back(&slot);
    }
    ~SlotOwner() {
        std::lock_guard lock(gSlotsMutex);
        std::erase(gSlots, &slot);
    }
};

std::string threadName(int tid) {
    char path[64], buf[32] = {};
    std::snprintf(path, sizeof path, "/proc/self/task/%d/comm", tid);
    FILE* f = std::fopen(path, "r");
    if (!f) return std::to_string(tid);
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    std::string s(buf, n);
    while (!s.empty() && s.back() == '\n') s.pop_back();
    return s;
}

// The return address the HLE call goes back to, then the first guest code addresses above it on
// the stack: the game function that waits and (likely) its callers.
struct StackScan {
    const uint64_t* stack;
    uint64_t found[3];
    int n;
};
void scanStack(void* p) {
    auto* s = static_cast<StackScan*>(p);
    // The stack belongs to a running thread: read each word once, so the address kept is the one
    // that was checked.
    for (int i = 0; i < 160 && s->n < 3; ++i) {
        const uint64_t a = reinterpret_cast<const volatile uint64_t*>(s->stack)[i];
        if (findModule(a)) s->found[s->n++] = a;
    }
}

std::string where(const uint64_t* stack) {
    StackScan s{stack, {}, 0};
    if (!stack || !guarded(scanStack, &s)) return {};
    std::string out;
    for (int i = 0; i < s.n; ++i) {
        Module* m = findModule(s.found[i]);
        if (!m) continue;
        std::string name = m->name();
        if (const size_t dot = name.find('.'); dot != std::string::npos) name.resize(dot);
        char cell[64];
        std::snprintf(cell, sizeof cell, "%s%s+0x%lx", i ? "<" : " @", name.c_str(), (unsigned long)(s.found[i] - uint64_t(m->base())));
        out += cell;
    }
    return out;
}

void waitSampler() {
    nameThread("lpn-waits");
    struct Seen {
        std::map<std::string, int> states;
        int samples = 0;
        uint32_t seqFrom = 0, seqTo = 0;
    };
    std::unordered_map<int, Seen> seen;
    auto next = std::chrono::steady_clock::now();
    auto report = next + std::chrono::seconds(1);
    IoCounters last{};
    for (;;) {
        next += std::chrono::milliseconds(5);
        std::this_thread::sleep_until(next);
        {
            std::lock_guard lock(gSlotsMutex);
            for (WaitSlot* s : gSlots) {
                const uint32_t a = s->seq.load(std::memory_order_acquire);
                std::string state = "not waiting";
                if (a & 1) {
                    const char* what = s->what.load(std::memory_order_relaxed);
                    std::string at = where(s->stack.load(std::memory_order_relaxed));
                    if (s->seq.load(std::memory_order_acquire) == a) state = std::string(what) + at;
                }
                auto [it, fresh] = seen.try_emplace(s->tid);
                Seen& t = it->second;
                if (fresh) t.seqFrom = a;
                t.seqTo = a;
                ++t.states[state];
                ++t.samples;
            }
        }
        if (next < report) continue;
        report += std::chrono::seconds(1);
        const double t = seconds();
        // Threads that sat in one wait the whole second are listed by name only.
        std::string asleep;
        std::vector<std::pair<std::string, std::string>> lines;
        for (auto& [tid, s] : seen) {
            const std::string name = threadName(tid);
            if (s.states.size() == 1 && s.seqFrom == s.seqTo && (s.seqFrom & 1)) {
                asleep += (asleep.empty() ? " " : ", ") + name;
                continue;
            }
            std::vector<std::pair<int, std::string>> rows;
            for (const auto& [state, n] : s.states) rows.push_back({n, state});
            std::sort(rows.rbegin(), rows.rend());
            std::string line;
            for (size_t i = 0; i < std::min<size_t>(rows.size(), 3); ++i) {
                const int pct = int(100.0 * rows[i].first / std::max(1, s.samples) + 0.5);
                if (i && pct < 5) break;
                line += (i ? ", " : " ") + rows[i].second + " " + std::to_string(pct) + "%";
            }
            line += " (" + std::to_string((s.seqTo - s.seqFrom) / 2) + " waits)";
            lines.push_back({name, line});
        }
        seen.clear();
        std::sort(lines.begin(), lines.end());
        for (const auto& [name, line] : lines) std::fprintf(stderr, "[wait] t=%.1f %s:%s\n", t, name.c_str(), line.c_str());
        if (!asleep.empty()) std::fprintf(stderr, "[wait] t=%.1f asleep all second:%s\n", t, asleep.c_str());
        IoCounters& c = io();
        auto delta = [](std::atomic<uint64_t>& now, std::atomic<uint64_t>& before) {
            const uint64_t v = now.load(std::memory_order_relaxed);
            const uint64_t d = v - before.load(std::memory_order_relaxed);
            before.store(v, std::memory_order_relaxed);
            return d;
        };
        const uint64_t opens = delta(c.opens, last.opens), reads = delta(c.reads, last.reads);
        const uint64_t bytes = delta(c.bytes, last.bytes), readNs = delta(c.readNs, last.readNs);
        const uint64_t inflates = delta(c.inflates, last.inflates), inflateNs = delta(c.inflateNs, last.inflateNs);
        if (opens || reads || inflates)
            std::fprintf(stderr, "[io] t=%.1f %llu opens, %llu reads %.1f MB in %.0f ms, %llu inflates in %.0f ms (flip %llu)\n", t,
                         (unsigned long long)opens, (unsigned long long)reads, bytes / 1e6, readNs / 1e6,
                         (unsigned long long)inflates, inflateNs / 1e6, (unsigned long long)flipCount());
    }
}

} // namespace

IoCounters& io() {
    static IoCounters c;
    return c;
}

bool on() {
    static const bool enabled = std::getenv("LPN_PROFILE") != nullptr;
    return enabled;
}

double seconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - gStart).count(); }

namespace {

// The recorder (lpn-record) caps the frame rate in crowds, alone at 100% of a core. Left to the
// scheduler it shared a core with a hashing thread (hyper-threading halves each) and ran on an
// E-core about a seventh of the time. It gets a P-core of its own: the fastest core's first
// logical CPU, the core's other logical CPUs left idle, and every other thread kept off that core.
// LPN_NO_PIN=1: no affinity set.
struct Cores {
    bool on = false;
    cpu_set_t recorder, others;
};
std::string readLine(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);
    return line;
}
// "0-1", "4,5": the CPUs listed.
std::vector<int> cpuList(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t end = s.find(',', i);
        if (end == std::string::npos) end = s.size();
        const std::string part = s.substr(i, end - i);
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str()), b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) out.push_back(c);
        i = end + 1;
    }
    return out;
}
const Cores& cores() {
    static const Cores c = [] {
        Cores c;
        if (getenv("LPN_NO_PIN")) return c;
        cpu_set_t allowed;
        if (sched_getaffinity(0, sizeof allowed, &allowed)) return c;
        const std::string sys = "/sys/devices/system/cpu/cpu";
        int best = -1;
        long bestFreq = -1;
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (!CPU_ISSET(i, &allowed)) continue;
            const long f = std::atol(readLine(sys + std::to_string(i) + "/cpufreq/cpuinfo_max_freq").c_str());
            if (f > bestFreq) best = i, bestFreq = f; // the first of the fastest
        }
        if (best < 0) return c;
        std::vector<int> core = cpuList(readLine(sys + std::to_string(best) + "/topology/thread_siblings_list"));
        if (core.empty()) core = {best};
        best = core.front();
        c.others = allowed;
        for (int s : core) CPU_CLR(s, &c.others);
        if (CPU_COUNT(&c.others) < 2) return c; // too few CPUs to set one core aside
        CPU_ZERO(&c.recorder);
        CPU_SET(best, &c.recorder);
        c.on = true;
        std::fprintf(stderr, "[cpu] the recorder runs on CPU %d (%ld MHz), its core's other CPUs idle (LPN_NO_PIN=1: no pinning)\n",
                     best, bestFreq / 1000);
        return c;
    }();
    return c;
}

} // namespace

void avoidRecorderCore() {
    if (cores().on) sched_setaffinity(0, sizeof(cpu_set_t), &cores().others);
}

void nameThread(const char* name) {
    pthread_setname_np(pthread_self(), name);
    if (!cores().on) return;
    if (!std::strcmp(name, "lpn-record")) sched_setaffinity(0, sizeof(cpu_set_t), &cores().recorder);
    else avoidRecorderCore(); // also for threads the recorder starts (they inherit its CPU)
}

void mark() {
    const int n = ++gMarks;
    std::fprintf(stderr, "[mark] %d at t=%.1f%s\n", n, seconds(), on() ? "" : " (LPN_PROFILE is off: no fps reports)");
    if (on()) startTimeline(0.25); // what each thread does over a few frames from here ([tl] lines)
}

void start() {
    if (!on()) return;
    std::thread(sampler).detach();
    std::thread(waitSampler).detach();
}

} // namespace host::profile

namespace host {

bool waitSamplingOn() { return profile::on(); }

WaitSlot& waitSlot() {
    thread_local profile::SlotOwner owner;
    return owner.slot;
}

} // namespace host
