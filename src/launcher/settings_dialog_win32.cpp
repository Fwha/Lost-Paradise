// The Settings window on Windows (settings_dialog.h): a window of the system's common controls
// (a tab control, drop-down lists, checkboxes and push buttons) owned by the launcher's window,
// which is disabled while it is open, as a modal dialog's owner is. The manifest
// (packaging/windows/app.manifest) asks for version 6 of the controls, the themed ones, and
// per-monitor DPI: sizes here are at 96 DPI and scaled to the monitor's.
#include "launcher/settings_dialog.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commctrl.h>

#include <string>

namespace launcher::settings {

namespace {

constexpr int kDefaultsId = 100, kTabId = 101, kFirstControlId = 200;
// The layout at 96 DPI.
constexpr int kClientW = 604, kBannerH = 80, kMargin = 10, kTabH = 330, kRowH = 28, kLabelW = 180, kControlW = 240;
constexpr int kButtonW = 130, kButtonH = 26;

std::wstring wide(const std::string& s) {
    std::wstring w(size_t(MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0)), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), w.data(), int(w.size()));
    return w;
}

struct Row {
    const Option* option;
    int tab;
    HWND label = nullptr, control = nullptr;
};

struct State {
    HWND wnd = nullptr, tabCtl = nullptr;
    std::vector<Row> rows;
    std::vector<std::pair<int, HWND>> notes; // each tab's note line
    HBRUSH page = nullptr;                   // the tab pages' background, under their labels
    int result = 0;                          // IDOK or IDCANCEL once closed
};

void fill(const Row& r, const host::DisplaySettings& s) {
    if (r.option->checkbox()) {
        SendMessageW(r.control, BM_SETCHECK, r.option->get(s) ? BST_CHECKED : BST_UNCHECKED, 0);
        return;
    }
    SendMessageW(r.control, CB_RESETCONTENT, 0, 0);
    for (const std::string& c : r.option->choices(s))
        SendMessageW(r.control, CB_ADDSTRING, 0, LPARAM(wide(c).c_str()));
    SendMessageW(r.control, CB_SETCURSEL, WPARAM(r.option->get(s)), 0);
}

void showTab(State& st, int tab) {
    for (const Row& r : st.rows) {
        if (r.label) ShowWindow(r.label, r.tab == tab ? SW_SHOW : SW_HIDE);
        ShowWindow(r.control, r.tab == tab ? SW_SHOW : SW_HIDE);
    }
    for (const auto& [t, note] : st.notes) ShowWindow(note, t == tab ? SW_SHOW : SW_HIDE);
}

int gLastTab = 0; // the tab the window opens on: the one it was closed on

LRESULT CALLBACK wndProc(HWND wnd, UINT msg, WPARAM wp, LPARAM lp) {
    State* st = reinterpret_cast<State*>(GetWindowLongPtrW(wnd, GWLP_USERDATA));
    switch (msg) {
    case WM_NCCREATE:
        SetWindowLongPtrW(wnd, GWLP_USERDATA, LONG_PTR(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams));
        break;
    case DM_GETDEFID: return MAKELRESULT(IDOK, DC_HASDEFID); // Enter presses Ok
    case WM_COMMAND:
        if (!st) break;
        if (LOWORD(wp) == IDOK || LOWORD(wp) == IDCANCEL) st->result = LOWORD(wp);
        else if (LOWORD(wp) == kDefaultsId)
            for (const Row& r : st->rows) fill(r, host::DisplaySettings{});
        return 0;
    case WM_NOTIFY:
        if (st && reinterpret_cast<NMHDR*>(lp)->idFrom == kTabId && reinterpret_cast<NMHDR*>(lp)->code == TCN_SELCHANGE)
            showTab(*st, int(SendMessageW(st->tabCtl, TCM_GETCURSEL, 0, 0)));
        return 0;
    case WM_CTLCOLORSTATIC:
        // Labels and checkboxes on a tab page take the page's colour; the banner and buttons don't.
        if (st && st->page && GetDlgCtrlID(HWND(lp)) >= kFirstControlId) {
            SetBkColor(HDC(wp), GetSysColor(COLOR_WINDOW));
            return LRESULT(st->page);
        }
        break;
    case WM_CLOSE:
        if (st) st->result = IDCANCEL;
        return 0;
    }
    return DefWindowProcW(wnd, msg, wp, lp);
}

UINT dpiOf(HWND wnd) {
    using GetDpiForWindowFn = UINT(WINAPI*)(HWND);
    static const auto fn = reinterpret_cast<GetDpiForWindowFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"user32.dll"), "GetDpiForWindow")));
    const UINT dpi = fn && wnd ? fn(wnd) : 0;
    if (dpi) return dpi;
    HDC dc = GetDC(nullptr);
    const UINT d = UINT(GetDeviceCaps(dc, LOGPIXELSY));
    ReleaseDC(nullptr, dc);
    return d ? d : 96;
}

HBITMAP bannerBitmap(const Image& art, int w, int h) {
    const Image b = coverCrop(art, w, h);
    if (b.rgba.empty()) return nullptr;
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof bi.bmiHeader;
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h; // top-down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(nullptr, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!bmp) return nullptr;
    auto* px = static_cast<uint8_t*>(bits);
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        px[i * 4 + 0] = b.rgba[i * 4 + 2];
        px[i * 4 + 1] = b.rgba[i * 4 + 1];
        px[i * 4 + 2] = b.rgba[i * 4 + 0];
        px[i * 4 + 3] = 255;
    }
    return bmp;
}

} // namespace

Result show(host::DisplaySettings& s, void* parent, const Image& art, const std::function<void()>& idle) {
    HWND owner = static_cast<HWND>(parent);
    HINSTANCE inst = GetModuleHandleW(nullptr);
    INITCOMMONCONTROLSEX icc{};
    icc.dwSize = sizeof icc;
    icc.dwICC = ICC_TAB_CLASSES | ICC_STANDARD_CLASSES | ICC_WIN95_CLASSES;
    InitCommonControlsEx(&icc);
    static const ATOM cls = [&] {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof wc;
        wc.lpfnWndProc = wndProc;
        wc.hInstance = inst;
        wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
        wc.hbrBackground = GetSysColorBrush(COLOR_BTNFACE);
        wc.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(1));
        wc.lpszClassName = L"LostParadiseSettings";
        return RegisterClassExW(&wc);
    }();
    if (!cls) return Result::Unavailable;

    const UINT dpi = dpiOf(owner);
    auto px = [dpi](int v) { return MulDiv(v, int(dpi), 96); };
    const bool hasBanner = !art.rgba.empty();
    const int bannerH = hasBanner ? kBannerH : 0;
    const int tabY = bannerH + kMargin, buttonsY = tabY + kTabH + kMargin;
    const int clientH = buttonsY + kButtonH + kMargin;

    State st;
    const DWORD style = WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_CLIPCHILDREN;
    const DWORD exStyle = WS_EX_DLGMODALFRAME | WS_EX_CONTROLPARENT;
    RECT rc{0, 0, px(kClientW), px(clientH)};
    AdjustWindowRectEx(&rc, style, FALSE, exStyle);
    const int ww = rc.right - rc.left, wh = rc.bottom - rc.top;
    // Centred on the launcher's window (or the screen).
    RECT over{0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
    if (owner) GetWindowRect(owner, &over);
    st.wnd = CreateWindowExW(exStyle, L"LostParadiseSettings", L"Lost Paradise - Settings", style,
                             (over.left + over.right - ww) / 2, (over.top + over.bottom - wh) / 2, ww, wh, owner,
                             nullptr, inst, &st);
    if (!st.wnd) return Result::Unavailable;

    NONCLIENTMETRICSW ncm{};
    ncm.cbSize = sizeof ncm;
    SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof ncm, &ncm, 0);
    LOGFONTW lf = ncm.lfMessageFont;
    lf.lfHeight = -MulDiv(9, int(dpi), 72);
    HFONT font = CreateFontIndirectW(&lf);
    st.page = CreateSolidBrush(GetSysColor(COLOR_WINDOW));

    int nextId = kFirstControlId;
    auto make = [&](const wchar_t* cls, const std::wstring& text, DWORD style, int x, int y, int w, int h, int id) {
        HWND c = CreateWindowExW(0, cls, text.c_str(), WS_CHILD | style, px(x), px(y), px(w), px(h), st.wnd,
                                 reinterpret_cast<HMENU>(INT_PTR(id)), inst, nullptr);
        SendMessageW(c, WM_SETFONT, WPARAM(font), TRUE);
        return c;
    };

    HBITMAP banner = hasBanner ? bannerBitmap(art, px(kClientW), px(kBannerH)) : nullptr;
    if (banner) {
        HWND b = make(L"STATIC", L"", WS_VISIBLE | SS_BITMAP, 0, 0, kClientW, kBannerH, -1);
        SendMessageW(b, STM_SETIMAGE, IMAGE_BITMAP, LPARAM(banner));
    }

    st.tabCtl = make(WC_TABCONTROLW, L"", WS_VISIBLE | WS_CLIPSIBLINGS | WS_TABSTOP, kMargin, tabY,
                     kClientW - 2 * kMargin, kTabH, kTabId);
    HWND tips = CreateWindowExW(WS_EX_TOPMOST, TOOLTIPS_CLASSW, nullptr, WS_POPUP | TTS_ALWAYSTIP | TTS_NOPREFIX,
                                CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, st.wnd, nullptr, inst, nullptr);
    SendMessageW(tips, TTM_SETMAXTIPWIDTH, 0, px(360));
    std::vector<std::wstring> tipTexts; // alive while the tooltips are
    for (const Tab& t : tabs()) for (const Option& o : t.options) tipTexts.push_back(wide(o.tip));
    size_t tipIndex = 0;
    auto tip = [&](HWND c, const std::wstring& text) {
        TOOLINFOW ti{};
        ti.cbSize = sizeof ti;
        ti.uFlags = TTF_IDISHWND | TTF_SUBCLASS;
        ti.hwnd = st.wnd;
        ti.uId = UINT_PTR(c);
        ti.lpszText = const_cast<wchar_t*>(text.c_str());
        SendMessageW(tips, TTM_ADDTOOLW, 0, LPARAM(&ti));
    };

    const int x0 = kMargin + 14, y0 = tabY + 36, cx = x0 + kLabelW;
    for (int ti = 0; ti < int(tabs().size()); ++ti) {
        const Tab& t = tabs()[size_t(ti)];
        std::wstring name = wide(t.name);
        TCITEMW item{};
        item.mask = TCIF_TEXT;
        item.pszText = name.data();
        SendMessageW(st.tabCtl, TCM_INSERTITEMW, WPARAM(ti), LPARAM(&item));
        int y = y0;
        for (const Option& o : t.options) {
            Row r{&o, ti};
            if (o.checkbox()) {
                r.control = make(L"BUTTON", wide(o.label), BS_AUTOCHECKBOX | WS_TABSTOP, cx, y + 2, kControlW, 20, nextId++);
            } else {
                r.label = make(L"STATIC", wide(o.label), SS_LEFT | SS_NOTIFY, x0, y + 4, kLabelW - 10, 20, nextId++);
                r.control = make(WC_COMBOBOXW, L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, cx, y, kControlW, 200, nextId++);
                tip(r.label, tipTexts[tipIndex]);
            }
            tip(r.control, tipTexts[tipIndex++]);
            fill(r, s);
            st.rows.push_back(r);
            y += kRowH;
        }
        if (const std::string note = tabNote(t); !note.empty())
            st.notes.push_back({ti, make(L"STATIC", wide(note), SS_LEFT, x0, tabY + kTabH - 46, kClientW - 2 * x0, 36, nextId++)});
    }
    // The tab control under the controls on its pages.
    SetWindowPos(st.tabCtl, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);

    make(L"BUTTON", L"Set To Default", WS_VISIBLE | WS_TABSTOP | BS_PUSHBUTTON, kMargin, buttonsY, kButtonW, kButtonH, kDefaultsId);
    HWND ok = make(L"BUTTON", L"Ok", WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON, kClientW - kMargin - 2 * kButtonW - 8, buttonsY,
                   kButtonW, kButtonH, IDOK);
    make(L"BUTTON", L"Cancel", WS_VISIBLE | WS_TABSTOP | BS_PUSHBUTTON, kClientW - kMargin - kButtonW, buttonsY, kButtonW,
         kButtonH, IDCANCEL);

    const int tabCount = int(tabs().size());
    const int first = gLastTab < tabCount ? gLastTab : 0;
    SendMessageW(st.tabCtl, TCM_SETCURSEL, WPARAM(first), 0);
    showTab(st, first);

    if (owner) EnableWindow(owner, FALSE);
    ShowWindow(st.wnd, SW_SHOW);
    SetFocus(ok);
    // The dialog's own message loop, which also delivers the launcher's window messages (SDL
    // reads them into its queue, and idle keeps that window drawn).
    while (!st.result) {
        MsgWaitForMultipleObjects(0, nullptr, FALSE, 15, QS_ALLINPUT);
        MSG m;
        while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) {
            if (m.message == WM_QUIT) {
                PostQuitMessage(int(m.wParam));
                st.result = IDCANCEL;
                break;
            }
            // Ctrl+Tab and Ctrl+Shift+Tab change tabs, as in the system's own dialogs.
            if (m.message == WM_KEYDOWN && m.wParam == VK_TAB && (GetKeyState(VK_CONTROL) & 0x8000)) {
                const int n = int(SendMessageW(st.tabCtl, TCM_GETCURSEL, 0, 0));
                const int next = (n + ((GetKeyState(VK_SHIFT) & 0x8000) ? tabCount - 1 : 1)) % tabCount;
                SendMessageW(st.tabCtl, TCM_SETCURSEL, WPARAM(next), 0);
                showTab(st, next);
                continue;
            }
            if (!IsDialogMessageW(st.wnd, &m)) {
                TranslateMessage(&m);
                DispatchMessageW(&m);
            }
        }
        if (idle) idle();
    }
    if (st.result == IDOK)
        for (const Row& r : st.rows) {
            const int v = r.option->checkbox() ? int(SendMessageW(r.control, BM_GETCHECK, 0, 0) == BST_CHECKED)
                                               : int(SendMessageW(r.control, CB_GETCURSEL, 0, 0));
            if (v >= 0) r.option->set(s, v);
        }
    gLastTab = int(SendMessageW(st.tabCtl, TCM_GETCURSEL, 0, 0));
    // Give the launcher's window back its input before this one goes, so it is the one activated.
    if (owner) EnableWindow(owner, TRUE);
    DestroyWindow(st.wnd);
    if (owner) SetForegroundWindow(owner);
    DeleteObject(font);
    DeleteObject(st.page);
    if (banner) DeleteObject(banner);
    return st.result == IDOK ? Result::Ok : Result::Cancel;
}

} // namespace launcher::settings
