#include "host/profile.h"

#include <dirent.h>
#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

} // namespace

bool on() {
    static const bool enabled = std::getenv("LPN_PROFILE") != nullptr;
    return enabled;
}

double seconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - gStart).count(); }

void nameThread(const char* name) { pthread_setname_np(pthread_self(), name); }

void mark() {
    const int n = ++gMarks;
    std::fprintf(stderr, "[mark] %d at t=%.1f%s\n", n, seconds(), on() ? "" : " (LPN_PROFILE is off: no fps reports)");
}

void start() {
    if (on()) std::thread(sampler).detach();
}

} // namespace host::profile
