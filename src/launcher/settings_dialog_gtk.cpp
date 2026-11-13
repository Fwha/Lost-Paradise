// The Settings window on Linux (settings_dialog.h): GTK 3, the toolkit of most desktops' own
// dialogs. It is loaded with dlopen when the window first opens rather than linked, so the
// launcher (and the AppImage) needs no GTK to build or start: without it, show() returns
// Unavailable and the launcher keeps its own Settings screen.
#include "launcher/settings_dialog.h"

#include <dlfcn.h>

#include <cstdio>

namespace launcher::settings {

namespace {

using W = void*; // GtkWidget*, GdkPixbuf* and the other GObjects, opaque here
using gboolean = int;

// The calls used, by name. Every one of them has been in GTK since 3.16 at the latest.
#define LPN_GTK_CALLS(X)                                                                                   \
    X(void, gtk_disable_setlocale, ())                                                                     \
    X(gboolean, gtk_init_check, (int*, char***))                                                           \
    X(W, gtk_dialog_new, ())                                                                               \
    X(W, gtk_dialog_add_button, (W, const char*, int))                                                     \
    X(W, gtk_dialog_get_content_area, (W))                                                                 \
    X(W, gtk_dialog_get_action_area, (W))                                                                  \
    X(void, gtk_dialog_set_default_response, (W, int))                                                     \
    X(void, gtk_button_box_set_child_secondary, (W, W, gboolean))                                          \
    X(void, gtk_window_set_title, (W, const char*))                                                        \
    X(void, gtk_window_set_modal, (W, gboolean))                                                           \
    X(void, gtk_window_set_resizable, (W, gboolean))                                                       \
    X(void, gtk_window_set_position, (W, int))                                                             \
    X(void, gtk_window_set_keep_above, (W, gboolean))                                                      \
    X(void, gtk_window_present, (W))                                                                       \
    X(W, gtk_notebook_new, ())                                                                             \
    X(int, gtk_notebook_append_page, (W, W, W))                                                            \
    X(void, gtk_notebook_set_current_page, (W, int))                                                       \
    X(int, gtk_notebook_get_current_page, (W))                                                             \
    X(W, gtk_box_new, (int, int))                                                                          \
    X(void, gtk_box_pack_start, (W, W, gboolean, gboolean, unsigned))                                      \
    X(void, gtk_box_pack_end, (W, W, gboolean, gboolean, unsigned))                                        \
    X(W, gtk_grid_new, ())                                                                                 \
    X(void, gtk_grid_attach, (W, W, int, int, int, int))                                                   \
    X(void, gtk_grid_set_row_spacing, (W, unsigned))                                                       \
    X(void, gtk_grid_set_column_spacing, (W, unsigned))                                                    \
    X(W, gtk_label_new, (const char*))                                                                     \
    X(void, gtk_label_set_xalign, (W, float))                                                              \
    X(void, gtk_label_set_line_wrap, (W, gboolean))                                                        \
    X(void, gtk_label_set_max_width_chars, (W, int))                                                       \
    X(W, gtk_combo_box_text_new, ())                                                                       \
    X(void, gtk_combo_box_text_append_text, (W, const char*))                                              \
    X(void, gtk_combo_box_text_remove_all, (W))                                                            \
    X(void, gtk_combo_box_set_active, (W, int))                                                            \
    X(int, gtk_combo_box_get_active, (W))                                                                  \
    X(W, gtk_check_button_new_with_label, (const char*))                                                   \
    X(void, gtk_toggle_button_set_active, (W, gboolean))                                                   \
    X(gboolean, gtk_toggle_button_get_active, (W))                                                         \
    X(W, gtk_image_new_from_pixbuf, (W))                                                                   \
    X(void, gtk_widget_set_tooltip_text, (W, const char*))                                                 \
    X(void, gtk_widget_set_size_request, (W, int, int))                                                    \
    X(void, gtk_widget_set_hexpand, (W, gboolean))                                                         \
    X(void, gtk_widget_set_margin_start, (W, int))                                                         \
    X(void, gtk_widget_set_margin_end, (W, int))                                                           \
    X(void, gtk_widget_set_margin_top, (W, int))                                                           \
    X(void, gtk_widget_set_margin_bottom, (W, int))                                                        \
    X(void, gtk_container_set_border_width, (W, unsigned))                                                \
    X(void, gtk_widget_show_all, (W))                                                                      \
    X(void, gtk_widget_destroy, (W))                                                                       \
    X(gboolean, gtk_events_pending, ())                                                                    \
    X(gboolean, gtk_main_iteration_do, (gboolean))                                                         \
    X(W, gdk_pixbuf_new_from_data, (const unsigned char*, int, gboolean, int, int, int, int, void*, void*)) \
    X(void, g_object_unref, (W))                                                                           \
    X(unsigned long, g_signal_connect_data, (W, const char*, void (*)(), void*, void*, int))

struct Gtk {
#define LPN_GTK_FIELD(ret, name, args) ret(*name) args = nullptr;
    LPN_GTK_CALLS(LPN_GTK_FIELD)
#undef LPN_GTK_FIELD
};

// Loaded and started once; nullptr when there is no GTK 3 or no display for it.
const Gtk* gtk() {
    static Gtk g;
    static const bool ok = [] {
        void* lib = dlopen("libgtk-3.so.0", RTLD_NOW | RTLD_GLOBAL);
        if (!lib) {
            std::fprintf(stderr, "[launcher] no GTK 3 (%s): using the built-in Settings screen\n", dlerror());
            return false;
        }
#define LPN_GTK_LOAD(ret, name, args)                                                     \
    g.name = reinterpret_cast<ret(*) args>(dlsym(lib, #name));                            \
    if (!g.name) {                                                                        \
        std::fprintf(stderr, "[launcher] GTK 3 has no %s: using the built-in Settings screen\n", #name); \
        return false;                                                                     \
    }
        LPN_GTK_CALLS(LPN_GTK_LOAD)
#undef LPN_GTK_LOAD
        // GTK would otherwise set the C locale from the environment, and the launcher writes the
        // ini with printf.
        g.gtk_disable_setlocale();
        if (!g.gtk_init_check(nullptr, nullptr)) {
            std::fprintf(stderr, "[launcher] GTK 3 cannot open the display: using the built-in Settings screen\n");
            return false;
        }
        return true;
    }();
    return ok ? &g : nullptr;
}

constexpr int kOk = -5, kCancel = -6, kDeleteEvent = -4, kDefaults = 1; // GTK_RESPONSE_*, ours
constexpr int kVertical = 1, kCenter = 1, kRgb = 0;                       // GTK_ORIENTATION_VERTICAL, ...
constexpr int kBannerW = 600, kBannerH = 80;

struct Row {
    const Option* option;
    W control; // a GtkComboBoxText or a GtkCheckButton
};

void fill(const Gtk& g, const Row& r, const host::DisplaySettings& s) {
    if (r.option->checkbox()) {
        g.gtk_toggle_button_set_active(r.control, r.option->get(s) != 0);
        return;
    }
    g.gtk_combo_box_text_remove_all(r.control);
    for (const std::string& c : r.option->choices(s)) g.gtk_combo_box_text_append_text(r.control, c.c_str());
    g.gtk_combo_box_set_active(r.control, r.option->get(s));
}

void onResponse(W, int response, void* out) { *static_cast<int*>(out) = response; }
// The close button and Escape: kept from destroying the window, which is destroyed below.
gboolean onDelete(W, void*, void* out) {
    *static_cast<int*>(out) = kDeleteEvent;
    return true;
}

int gLastTab = 0; // the tab the window opens on: the one it was closed on

} // namespace

Result show(host::DisplaySettings& s, void*, const Image& art, const std::function<void()>& idle) {
    const Gtk* gp = gtk();
    if (!gp) return Result::Unavailable;
    const Gtk& g = *gp;

    W dialog = g.gtk_dialog_new();
    g.gtk_window_set_title(dialog, "Lost Paradise - Settings");
    g.gtk_window_set_modal(dialog, true);
    g.gtk_window_set_resizable(dialog, false);
    g.gtk_window_set_position(dialog, kCenter);
    // Not a child of the launcher's (SDL) window, which GTK cannot name: kept in front of it.
    g.gtk_window_set_keep_above(dialog, true);
    W content = g.gtk_dialog_get_content_area(dialog);
    // The banner from edge to edge; the tabs and the buttons inset from them.
    g.gtk_container_set_border_width(dialog, 0);
    g.gtk_container_set_border_width(content, 0);
    g.gtk_container_set_border_width(g.gtk_dialog_get_action_area(dialog), 10);

    Image banner;
    if (!art.rgba.empty()) banner = coverCrop(art, kBannerW, kBannerH);
    if (!banner.rgba.empty()) {
        W pix = g.gdk_pixbuf_new_from_data(banner.rgba.data(), kRgb, true, 8, banner.w, banner.h, banner.w * 4, nullptr, nullptr);
        g.gtk_box_pack_start(content, g.gtk_image_new_from_pixbuf(pix), false, false, 0);
        g.g_object_unref(pix); // the image holds its own reference
    }

    W notebook = g.gtk_notebook_new();
    g.gtk_widget_set_margin_start(notebook, 10);
    g.gtk_widget_set_margin_end(notebook, 10);
    g.gtk_widget_set_margin_top(notebook, 10);
    g.gtk_box_pack_start(content, notebook, true, true, 0);

    std::vector<Row> rows;
    for (const Tab& t : tabs()) {
        W page = g.gtk_box_new(kVertical, 12);
        g.gtk_widget_set_size_request(page, kBannerW - 34, 300);
        W grid = g.gtk_grid_new();
        g.gtk_grid_set_row_spacing(grid, 10);
        g.gtk_grid_set_column_spacing(grid, 24);
        g.gtk_widget_set_margin_start(page, 14);
        g.gtk_widget_set_margin_end(page, 14);
        g.gtk_widget_set_margin_top(page, 14);
        g.gtk_widget_set_margin_bottom(page, 12);
        g.gtk_box_pack_start(page, grid, false, false, 0);
        int y = 0;
        for (const Option& o : t.options) {
            W control;
            if (o.checkbox()) {
                control = g.gtk_check_button_new_with_label(o.label);
                g.gtk_grid_attach(grid, control, 1, y, 1, 1);
            } else {
                W label = g.gtk_label_new(o.label);
                g.gtk_label_set_xalign(label, 0);
                g.gtk_widget_set_tooltip_text(label, o.tip);
                g.gtk_grid_attach(grid, label, 0, y, 1, 1);
                control = g.gtk_combo_box_text_new();
                g.gtk_widget_set_size_request(control, 260, -1);
                g.gtk_grid_attach(grid, control, 1, y, 1, 1);
            }
            g.gtk_widget_set_tooltip_text(control, o.tip);
            rows.push_back({&o, control});
            fill(g, rows.back(), s);
            ++y;
        }
        // Keep the label column as wide on every tab, so the controls line up between them.
        W spacer = g.gtk_label_new("");
        g.gtk_widget_set_size_request(spacer, 170, -1);
        g.gtk_grid_attach(grid, spacer, 0, y, 1, 1);
        if (const std::string note = tabNote(t); !note.empty()) {
            W label = g.gtk_label_new(note.c_str());
            g.gtk_label_set_xalign(label, 0);
            g.gtk_label_set_line_wrap(label, true);
            g.gtk_label_set_max_width_chars(label, 70);
            g.gtk_box_pack_end(page, label, false, false, 0);
        }
        g.gtk_notebook_append_page(notebook, page, g.gtk_label_new(t.name));
    }

    W defaults = g.gtk_dialog_add_button(dialog, "Set To Default", kDefaults);
    g.gtk_button_box_set_child_secondary(g.gtk_dialog_get_action_area(dialog), defaults, true);
    g.gtk_dialog_add_button(dialog, "Ok", kOk);
    g.gtk_dialog_add_button(dialog, "Cancel", kCancel);
    g.gtk_dialog_set_default_response(dialog, kOk);

    int response = 0;
    g.g_signal_connect_data(dialog, "response", reinterpret_cast<void (*)()>(&onResponse), &response, nullptr, 0);
    g.g_signal_connect_data(dialog, "delete-event", reinterpret_cast<void (*)()>(&onDelete), &response, nullptr, 0);
    g.gtk_widget_show_all(dialog);
    g.gtk_notebook_set_current_page(notebook, gLastTab);
    g.gtk_window_present(dialog);

    // Run GTK's loop and the launcher's together, so its window stays drawn and answers the
    // desktop's are-you-alive checks while this one is open.
    for (;;) {
        while (g.gtk_events_pending()) g.gtk_main_iteration_do(false);
        if (response == kDefaults) {
            response = 0;
            const host::DisplaySettings d{};
            for (const Row& r : rows) fill(g, r, d);
        }
        if (response == kOk || response == kCancel || response == kDeleteEvent) break;
        if (idle) idle();
    }
    if (response == kOk)
        for (const Row& r : rows) {
            const int v = r.option->checkbox() ? int(g.gtk_toggle_button_get_active(r.control) != 0)
                                               : g.gtk_combo_box_get_active(r.control);
            if (v >= 0) r.option->set(s, v);
        }
    gLastTab = g.gtk_notebook_get_current_page(notebook);
    g.gtk_widget_destroy(dialog);
    while (g.gtk_events_pending()) g.gtk_main_iteration_do(false);
    return response == kOk ? Result::Ok : Result::Cancel;
}

} // namespace launcher::settings
