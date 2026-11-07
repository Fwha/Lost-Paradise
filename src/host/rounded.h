// Rounded rectangles for our own screens (the menus drawn over the game and the launcher), drawn
// as triangles with SDL_RenderGeometry in the renderer's blend mode.
#pragma once

#include <SDL.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ui {

enum Corners : unsigned {
    kTopLeft = 1, kTopRight = 2, kBottomRight = 4, kBottomLeft = 8,
    kTop = kTopLeft | kTopRight, kBottom = kBottomRight | kBottomLeft, kLeft = kTopLeft | kBottomLeft,
    kAll = kTop | kBottom,
};

// The edge of a rounded rectangle, clockwise from the top left, the same number of points for
// every radius (a square corner repeats its point) so two of them can be stitched into a ring.
inline void roundedPath(std::vector<SDL_FPoint>& p, float x, float y, float w, float h, float r, unsigned corners) {
    constexpr int kSteps = 8;
    constexpr float kPi = 3.14159265f;
    r = std::clamp(r, 0.0f, std::min(w, h) / 2);
    const float cx[4] = {x + r, x + w - r, x + w - r, x + r}, cy[4] = {y + r, y + r, y + h - r, y + h - r};
    const float ex[4] = {x, x + w, x + w, x}, ey[4] = {y, y, y + h, y + h};
    p.clear();
    for (int c = 0; c < 4; ++c) {
        const bool round = (corners & (1u << c)) && r > 0;
        const float a0 = kPi * (1.0f + 0.5f * c);
        for (int i = 0; i <= kSteps; ++i) {
            const float a = a0 + kPi / 2 * float(i) / kSteps;
            p.push_back(round ? SDL_FPoint{cx[c] + r * std::cos(a), cy[c] + r * std::sin(a)} : SDL_FPoint{ex[c], ey[c]});
        }
    }
}

inline void fillRounded(SDL_Renderer* ren, int x, int y, int w, int h, int r, SDL_Color c, unsigned corners = kAll) {
    if (w <= 0 || h <= 0) return;
    std::vector<SDL_FPoint> p;
    roundedPath(p, float(x), float(y), float(w), float(h), float(r), corners);
    std::vector<SDL_Vertex> v;
    std::vector<int> idx;
    v.push_back({{x + w / 2.0f, y + h / 2.0f}, c, {0, 0}});
    for (const SDL_FPoint& q : p) v.push_back({q, c, {0, 0}});
    const int n = int(p.size());
    for (int i = 0; i < n; ++i) idx.insert(idx.end(), {0, 1 + i, 1 + (i + 1) % n});
    SDL_RenderGeometry(ren, nullptr, v.data(), int(v.size()), idx.data(), int(idx.size()));
}

// A t pixels wide border on the inside of the rounded rectangle.
inline void outlineRounded(SDL_Renderer* ren, int x, int y, int w, int h, int r, int t, SDL_Color c,
                           unsigned corners = kAll) {
    if (w <= 2 * t || h <= 2 * t) return fillRounded(ren, x, y, w, h, r, c, corners);
    std::vector<SDL_FPoint> o, in;
    roundedPath(o, float(x), float(y), float(w), float(h), float(r), corners);
    roundedPath(in, float(x + t), float(y + t), float(w - 2 * t), float(h - 2 * t), float(std::max(0, r - t)), corners);
    std::vector<SDL_Vertex> v;
    std::vector<int> idx;
    const int n = int(o.size());
    for (int i = 0; i < n; ++i) v.push_back({o[size_t(i)], c, {0, 0}}), v.push_back({in[size_t(i)], c, {0, 0}});
    for (int i = 0; i < n; ++i) {
        const int a = 2 * i, b = 2 * ((i + 1) % n);
        idx.insert(idx.end(), {a, b, a + 1, a + 1, b, b + 1});
    }
    SDL_RenderGeometry(ren, nullptr, v.data(), int(v.size()), idx.data(), int(idx.size()));
}

} // namespace ui
