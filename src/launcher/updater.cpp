#include "launcher/updater.h"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef LPN_VERSION
#define LPN_VERSION "dev"
#endif

namespace fs = std::filesystem;

namespace launcher {

namespace {

constexpr const char* kRepo = "Fwha/Lost-Paradise";

std::mutex gM;
UpdateStatus gStatus;
std::string gInstall, gAssetUrl, gUpdated; // gUpdated: the AppImage written, ready to start
std::atomic<bool> gBusy{false}, gRestart{false};

void set(UpdateState s, std::string message = {}) {
    std::lock_guard lock(gM);
    gStatus.state = s;
    gStatus.message = std::move(message);
}

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    return s.substr(i);
}

// Runs argv (no shell) and returns its stdout; *ok is whether it exited with 0.
std::string run(const std::vector<std::string>& args, bool* ok) {
    *ok = false;
    int out[2];
    if (pipe(out) != 0) return {};
    const pid_t pid = fork();
    if (pid < 0) {
        close(out[0]), close(out[1]);
        return {};
    }
    if (pid == 0) {
        dup2(out[1], 1);
        close(out[0]), close(out[1]);
        std::vector<char*> a;
        for (const auto& s : args) a.push_back(const_cast<char*>(s.c_str()));
        a.push_back(nullptr);
        execvp(a[0], a.data());
        _exit(127);
    }
    close(out[1]);
    std::string text;
    char buf[4096];
    for (ssize_t n; (n = read(out[0], buf, sizeof buf)) > 0;) text.append(buf, size_t(n));
    close(out[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    *ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
    return text;
}

std::string token() {
    for (const char* v : {"LPN_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"})
        if (const char* t = std::getenv(v); t && *t) return trim(t);
    if (FILE* f = std::fopen((fs::path(gInstall) / "github-token").c_str(), "r")) {
        char buf[512] = {};
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        std::fclose(f);
        if (n) return trim(std::string(buf, n));
    }
    bool ok;
    const std::string t = trim(run({"gh", "auth", "token"}, &ok));
    return ok ? t : std::string();
}

std::string repo() {
    const char* r = std::getenv("LPN_UPDATE_REPO");
    return r && *r ? r : kRepo;
}

// curl with the token in a header file (kept off the command line, where ps would show it).
// Returns the body (or nothing with -o) and sets *code to the HTTP status (0: no answer).
std::string curl(std::vector<std::string> args, int* code) {
    const std::string t = token();
    char hdr[] = "/tmp/lpn-gh-XXXXXX";
    int fd = -1;
    if (!t.empty() && (fd = mkstemp(hdr)) >= 0) {
        const std::string line = "Authorization: Bearer " + t + "\n";
        if (write(fd, line.data(), line.size()) != ssize_t(line.size())) {}
        close(fd);
        args.insert(args.begin(), {"-H", std::string("@") + hdr});
    }
    args.insert(args.begin(), {"curl", "-sSL", "--connect-timeout", "15", "-w", "\n%{http_code}", "-H",
                               "X-GitHub-Api-Version: 2022-11-28"});
    bool ok;
    std::string out = run(args, &ok);
    if (fd >= 0) unlink(hdr);
    const size_t nl = out.rfind('\n');
    *code = ok && nl != std::string::npos ? std::atoi(out.c_str() + nl + 1) : 0;
    if (nl != std::string::npos) out.resize(nl);
    return out;
}

std::string httpError(int code) {
    if (code == 0) return "No answer from GitHub: check the connection";
    if (code == 401) return "GitHub refused the token: sign in again with `gh auth login`";
    if (code == 404)
        return token().empty() ? "Cannot see the repo: sign in with `gh auth login` or put a token in github-token"
                               : "No release yet";
    if (code == 403 || code == 429) return "GitHub is limiting requests: try again later";
    return "GitHub answered " + std::to_string(code);
}

// The string value after "key": at or after from, or empty. *at gets where the value starts.
std::string jsonString(const std::string& j, const std::string& key, size_t from = 0, size_t* at = nullptr) {
    const size_t k = j.find("\"" + key + "\"", from);
    if (k == std::string::npos) return {};
    size_t i = j.find(':', k);
    if (i == std::string::npos) return {};
    i = j.find_first_not_of(" \t\r\n", i + 1);
    if (i == std::string::npos || j[i] != '"') return {};
    std::string v;
    for (++i; i < j.size() && j[i] != '"'; ++i) {
        if (j[i] == '\\' && i + 1 < j.size()) ++i;
        v += j[i];
    }
    if (at) *at = k;
    return v;
}

void check() {
    // LPN_UPDATE_URL: another place for the latest release's JSON (for testing, e.g. a file:// URL).
    const char* over = std::getenv("LPN_UPDATE_URL");
    int code = 0;
    const std::string j = curl({"-H", "Accept: application/vnd.github+json",
                                over && *over ? over : "https://api.github.com/repos/" + repo() + "/releases/latest"},
                               &code);
    if (code != 200 && !(over && *over && code == 0 && !j.empty())) {
        set(UpdateState::Failed, httpError(code));
        return;
    }
    const std::string tag = jsonString(j, "tag_name");
    if (tag.empty()) {
        set(UpdateState::Failed, "No release found");
        return;
    }
    // The release's AppImage: the asset whose name ends in .AppImage. Its "url" (the API's, which
    // works for private repos too) comes before its "name" in the asset object.
    std::string asset;
    const size_t assets = j.find("\"assets\"");
    for (size_t pos = assets; assets != std::string::npos;) {
        size_t at = 0;
        const std::string name = jsonString(j, "name", pos, &at);
        if (name.empty()) break;
        pos = at + 1;
        std::string lower = name;
        for (char& c : lower) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (lower.size() > 9 && lower.compare(lower.size() - 9, 9, ".appimage") == 0) {
            const size_t u = j.rfind("\"url\"", at);
            if (u != std::string::npos && u > assets) asset = jsonString(j, "url", u);
            break;
        }
    }
    {
        std::lock_guard lock(gM);
        gStatus.latest = tag;
        gAssetUrl = asset;
    }
    if (!newerVersion(tag, version())) set(UpdateState::UpToDate, std::string("Up to date (") + version() + ")");
    else set(UpdateState::Available, "Update " + tag + " is available");
}

void download() {
    const char* app = std::getenv("APPIMAGE");
    std::string url, tag;
    {
        std::lock_guard lock(gM);
        url = gAssetUrl, tag = gStatus.latest;
    }
    if (!app || !*app) {
        set(UpdateState::Available, tag + " is out: git pull in lost-paradise, then rebuild");
        return;
    }
    if (url.empty()) {
        set(UpdateState::Failed, tag + " has no AppImage attached");
        return;
    }
    const std::string part = std::string(app) + ".part";
    int code = 0;
    curl({"-H", "Accept: application/octet-stream", "-o", part, url}, &code);
    std::error_code ec;
    const bool fileUrl = url.rfind("file://", 0) == 0; // tests
    if ((code != 200 && !fileUrl) || fs::file_size(part, ec) < 1024 * 1024) {
        fs::remove(part, ec);
        set(UpdateState::Failed, code != 200 && !fileUrl ? httpError(code) : "The download failed");
        return;
    }
    chmod(part.c_str(), 0755);
    // The running AppImage stays open through its old inode, so it can be replaced under it.
    if (std::rename(part.c_str(), app) != 0) {
        fs::remove(part, ec);
        set(UpdateState::Failed, std::string("Cannot replace ") + app);
        return;
    }
    {
        std::lock_guard lock(gM);
        gUpdated = app;
    }
    set(UpdateState::Downloading, "Updated to " + tag + ", restarting");
    gRestart = true;
}

void start(void (*fn)(), UpdateState state, const char* message) {
    if (gBusy.exchange(true)) return;
    set(state, message);
    std::thread([fn] {
        fn();
        gBusy = false;
    }).detach();
}

std::vector<long> numbers(std::string v) {
    if (!v.empty() && (v[0] == 'v' || v[0] == 'V')) v.erase(0, 1);
    std::vector<long> out;
    size_t i = 0;
    while (i < v.size() && std::isdigit(static_cast<unsigned char>(v[i]))) {
        size_t n = 0;
        out.push_back(std::stol(v.substr(i), &n));
        i += n;
        if (i < v.size() && v[i] == '.') ++i;
        else break;
    }
    return out;
}

} // namespace

const char* version() { return LPN_VERSION; }

bool newerVersion(const std::string& a, const std::string& b) {
    const std::vector<long> x = numbers(a), y = numbers(b);
    if (x.empty()) return false;
    if (y.empty()) return true;
    return x > y;
}

void checkForUpdates(const std::string& installDir) {
    gInstall = installDir;
    start(&check, UpdateState::Checking, "Checking for updates...");
}

void applyUpdate() {
    if (updateStatus().state != UpdateState::Available) return;
    start(&download, UpdateState::Downloading, "Downloading the update...");
}

UpdateStatus updateStatus() {
    std::lock_guard lock(gM);
    return gStatus;
}

bool restartReady() { return gRestart; }

void restart(int argc, char** argv) {
    std::string app;
    {
        std::lock_guard lock(gM);
        app = gUpdated;
    }
    std::vector<char*> a{app.data()};
    for (int i = 1; i < argc; ++i) a.push_back(argv[i]);
    a.push_back(nullptr);
    execv(app.c_str(), a.data());
    std::fprintf(stderr, "[launcher] cannot start %s\n", app.c_str());
    std::exit(1);
}

} // namespace launcher
