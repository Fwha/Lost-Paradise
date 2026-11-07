// Check for updates: asks GitHub for the repo's latest release and, run from the AppImage,
// replaces the AppImage with the release's one and starts it again.
//
// The repo is private, so requests carry a GitHub token: $LPN_GITHUB_TOKEN, $GITHUB_TOKEN or
// $GH_TOKEN, else a github-token file in the install folder, else `gh auth token` (the GitHub CLI's
// login). Requests go through curl. $LPN_UPDATE_REPO names another owner/repo.
#pragma once

#include <string>

namespace launcher {

// The version this launcher was built as (LPN_VERSION: the release tag, or git describe).
const char* version();

enum class UpdateState { Idle, Checking, UpToDate, Available, Downloading, Failed };

struct UpdateStatus {
    UpdateState state = UpdateState::Idle;
    std::string latest;  // the latest release's tag
    std::string message; // a line to show: what failed, or what to do
};

// Both start work on a thread and return at once; poll updateStatus().
void checkForUpdates(const std::string& installDir);
// After Available. From the AppImage, downloads the new one over it; restartReady() then turns
// true. Elsewhere (a build tree) it only says to pull and rebuild.
void applyUpdate();
UpdateStatus updateStatus();
bool restartReady();
// Starts the updated AppImage in this process's place (call once the window is gone).
void restart(int argc, char** argv);

// Whether tag a is a later version than b ("v1.2.0" > "v1.1.9"); a version that is not numbers
// ("dev", a commit) is older than any release.
bool newerVersion(const std::string& a, const std::string& b);

} // namespace launcher
