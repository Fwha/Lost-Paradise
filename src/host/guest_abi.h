// The calling convention between the game's code and ours.
//
// PS4 code uses the System V x86-64 convention. That is Linux's own, but Windows uses the
// Microsoft one, so on Windows:
//  - every function of ours the game calls (HLE imports, hooks) goes through guestEntry<&fn>(),
//    a System V entry point that forwards to fn;
//  - hooks that look at their caller use guestHook<&fn>() and LPN_GUEST_RETURN_ADDRESS() /
//    LPN_GUEST_FRAME() in place of __builtin_return_address(0) / __builtin_frame_address(0);
//  - every pointer to game code we call is declared with LPN_GUEST, e.g.
//    `void (LPN_GUEST* fn)(void*)`, so the compiler calls it the System V way.
// On Linux all of these are no-ops. Our own functions are never System V themselves: GCC 13
// for Windows miscompiles System V functions with exception cleanups, so only the thunks are.
#pragma once

#if defined(_WIN32)
#define LPN_GUEST __attribute__((sysv_abi))
#define LPN_GUEST_RETURN_ADDRESS() (::host::tGuestCaller.ret)
#define LPN_GUEST_FRAME() (::host::tGuestCaller.frame)
#else
#define LPN_GUEST
#define LPN_GUEST_RETURN_ADDRESS() __builtin_return_address(0)
#define LPN_GUEST_FRAME() __builtin_frame_address(0)
#endif

#include <type_traits>

namespace host {

#if defined(_WIN32)
template <auto F> struct GuestThunk;
// F is called, not inlined: GCC 13 miscompiles (an internal error in the prologue code of a
// later function) when a System V function has Microsoft-convention code inlined into it.
template <typename R, typename... A, R (*F)(A...)> struct GuestThunk<F> {
    static LPN_GUEST R call(A... a) {
        R (*volatile f)(A...) = F;
        return f(a...);
    }
};
template <typename R, typename... A, R (*F)(A...) noexcept> struct GuestThunk<F> {
    static LPN_GUEST R call(A... a) {
        R (*volatile f)(A...) noexcept = F;
        return f(a...);
    }
};
// The game code that called the current hook (guestHook).
struct GuestCaller {
    void* ret;
    void* frame;
};
inline thread_local GuestCaller tGuestCaller{};

template <auto F> struct GuestHookThunk;
template <typename R, typename... A, R (*F)(A...)> struct GuestHookThunk<F> {
    static LPN_GUEST R call(A... a) {
        tGuestCaller = {__builtin_return_address(0), __builtin_frame_address(0)};
        R (*volatile f)(A...) = F;
        return f(a...);
    }
};
template <auto F> void* guestHook() { return reinterpret_cast<void*>(&GuestHookThunk<F>::call); }

// Data imports (a variable's address) are bound as they are.
template <auto F> void* guestEntry() {
    if constexpr (std::is_function_v<std::remove_pointer_t<decltype(F)>>)
        return reinterpret_cast<void*>(&GuestThunk<F>::call);
    else
        return (void*)(F);
}
#else
template <auto F> void* guestEntry() { return (void*)(F); }
template <auto F> void* guestHook() { return (void*)(F); }
#endif

} // namespace host
