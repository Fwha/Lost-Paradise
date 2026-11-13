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
#include <unistd.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/wait.h>
#endif

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
#ifdef _WIN32
std::string quoteArg(const std::string& a) {
    std::string q = "\"";
    size_t slashes = 0;
    for (char c : a) {
        if (c == '\\') {
            ++slashes;
            continue;
        }
        if (c == '"') q.append(slashes * 2 + 1, '\\');
        else q.append(slashes, '\\');
        slashes = 0;
        q += c;
    }
    q.append(slashes * 2, '\\');
    return q + "\"";
}

std::string run(const std::vector<std::string>& args, bool* ok) {
    *ok = false;
    std::string cmd;
    for (const auto& a : args) cmd += (cmd.empty() ? "" : " ") + quoteArg(a);
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE rd, wr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return {};
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        CloseHandle(rd), CloseHandle(wr);
        return {};
    }
    CloseHandle(wr);
    std::string text;
    char buf[4096];
    for (DWORD n; ReadFile(rd, buf, sizeof buf, &n, nullptr) && n > 0;) text.append(buf, n);
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess), CloseHandle(pi.hThread);
    *ok = code == 0;
    return text;
}
#else
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
#endif

std::string token() {
    for (const char* v : {"LPN_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"})
        if (const char* t = std::getenv(v); t && *t) return trim(t);
    if (FILE* f = std::fopen((fs::path(gInstall) / "github-token").string().c_str(), "r")) {
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
#ifdef _WIN32
    char dir[MAX_PATH + 1] = {}, file[MAX_PATH + 1] = {};
    GetTempPathA(sizeof dir, dir);
    std::string hdr;
    int fd = -1;
    if (!t.empty() && GetTempFileNameA(dir, "lpn", 0, file)) {
        hdr = file;
        if (FILE* f = std::fopen(file, "wb")) {
            std::fprintf(f, "Authorization: Bearer %s\n", t.c_str());
            std::fclose(f);
            fd = 0;
            args.insert(args.begin(), {"-H", "@" + hdr});
        }
    }
#else
    char hdr[] = "/tmp/lpn-gh-XXXXXX";
    int fd = -1;
    if (!t.empty() && (fd = mkstemp(hdr)) >= 0) {
        const std::string line = "Authorization: Bearer " + t + "\n";
        if (write(fd, line.data(), line.size()) != ssize_t(line.size())) {}
        close(fd);
        args.insert(args.begin(), {"-H", std::string("@") + hdr});
    }
#endif
    // --retry: GitHub sometimes answers 5xx for a moment (e.g. while a release is being published);
    // curl retries those (and 408/429) with backoff before giving up.
    args.insert(args.begin(), {"curl", "-sSL", "--connect-timeout", "15", "--retry", "3", "-w", "\n%{http_code}",
                               "-H", "X-GitHub-Api-Version: 2022-11-28"});
    bool ok;
    std::string out = run(args, &ok);
#ifdef _WIN32
    if (fd >= 0) DeleteFileA(hdr.c_str());
#else
    if (fd >= 0) unlink(hdr);
#endif
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
    if (code >= 500) return "GitHub is having trouble (" + std::to_string(code) + "): try again in a minute";
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
    // The release's AppImage: the asset whose name ends in .AppImage (on Windows, the one ending in
    // -windows.zip). Its "url" (the API's, which works for private repos too) comes before its
    // "name" in the asset object.
#ifdef _WIN32
    const std::string suffix = "-windows.zip";
#else
    const std::string suffix = ".appimage";
#endif
    std::string asset;
    const size_t assets = j.find("\"assets\"");
    for (size_t pos = assets; assets != std::string::npos;) {
        size_t at = 0;
        const std::string name = jsonString(j, "name", pos, &at);
        if (name.empty()) break;
        pos = at + 1;
        std::string lower = name;
        for (char& c : lower) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (lower.size() > suffix.size() && lower.compare(lower.size() - suffix.size(), suffix.size(), suffix) == 0) {
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

#ifdef _WIN32
// The running launcher and the DLLs it holds can't be overwritten, but they can be renamed.
bool replaceFile(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    if (fs::exists(to, ec)) {
        fs::path old = to;
        old += ".old";
        fs::remove(old, ec);
        fs::rename(to, old, ec);
        if (ec) return false;
    }
    fs::create_directories(to.parent_path(), ec);
    fs::rename(from, to, ec);
    return !ec;
}

void download() {
    std::string url, tag;
    {
        std::lock_guard lock(gM);
        url = gAssetUrl, tag = gStatus.latest;
    }
    if (url.empty()) {
        set(UpdateState::Failed, tag + " has no Windows zip attached");
        return;
    }
    const fs::path install = gInstall, zip = install / "update.zip.part", unpacked = install / "update.tmp";
    int code = 0;
    curl({"-H", "Accept: application/octet-stream", "-o", zip.string(), url}, &code);
    std::error_code ec;
    if (code != 200 || fs::file_size(zip, ec) < 1024 * 1024) {
        fs::remove(zip, ec);
        set(UpdateState::Failed, code != 200 ? httpError(code) : "The download failed");
        return;
    }
    // Windows 10 and later have bsdtar, which reads zips. The zip holds one folder.
    fs::remove_all(unpacked, ec);
    fs::create_directories(unpacked, ec);
    bool ok;
    run({"tar", "-xf", zip.string(), "-C", unpacked.string(), "--strip-components=1"}, &ok);
    fs::remove(zip, ec);
    if (!ok) {
        fs::remove_all(unpacked, ec);
        set(UpdateState::Failed, "Cannot unpack the update");
        return;
    }
    for (auto it = fs::recursive_directory_iterator(unpacked, ec); !ec && it != fs::recursive_directory_iterator();
         it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        const fs::path rel = fs::relative(it->path(), unpacked, ec);
        if (!replaceFile(it->path(), install / rel)) {
            set(UpdateState::Failed, "Cannot replace " + rel.string());
            return;
        }
    }
    fs::remove_all(unpacked, ec);
    wchar_t self[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(nullptr, self, DWORD(std::size(self)));
    {
        std::lock_guard lock(gM);
        gUpdated = fs::path(std::wstring(self, n)).string();
    }
    set(UpdateState::Downloading, "Updated to " + tag + ", restarting");
    gRestart = true;
}
#else
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
#endif

void start(void (*fn)(), UpdateState state, const char* message) {
    if (gBusy.exchange(true)) return;
    set(state, message);
    std::thread([fn] {
        fn();
        gBusy = false;
    }).detach();
}

// The numbers of a release tag (v1.2.3). Anything else (a CI build of a branch or pull request,
// e.g. "main" or "75-merge") has none, so it is never taken for a release and always offered one.
std::vector<long> numbers(std::string v) {
    if (v.size() < 2 || (v[0] != 'v' && v[0] != 'V') || !std::isdigit(static_cast<unsigned char>(v[1]))) return {};
    v.erase(0, 1);
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

#ifdef _WIN32
bool canSelfUpdate() { return true; }

void cleanUpAfterUpdate(const std::string& installDir) {
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(installDir, ec); !ec && it != fs::recursive_directory_iterator();
         it.increment(ec)) {
        if (it.depth() > 2) it.disable_recursion_pending();
        if (it->path().extension() == ".old" && it->is_regular_file()) fs::remove(it->path(), ec);
        const std::string n = it->path().filename().string();
        if (it->is_directory() && (n == "game" || n == "lpn-data")) it.disable_recursion_pending();
    }
}

void restart(int argc, char** argv) {
    std::string app;
    {
        std::lock_guard lock(gM);
        app = gUpdated;
    }
    std::string cmd = quoteArg(app);
    for (int i = 1; i < argc; ++i) cmd += " " + quoteArg(argv[i]);
    STARTUPINFOA si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (CreateProcessA(app.c_str(), cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        CloseHandle(pi.hThread), CloseHandle(pi.hProcess);
        std::exit(0);
    }
    std::fprintf(stderr, "[launcher] cannot start %s\n", app.c_str());
    std::exit(1);
}
#else
bool canSelfUpdate() { return std::getenv("APPIMAGE") != nullptr; }

void cleanUpAfterUpdate(const std::string&) {}

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
#endif

} // namespace launcher
