// libkernel threads and synchronisation: scePthread*, semaphores.
//
// Each PS4 handle (ScePthreadMutex, ScePthreadAttr, ...) is a pointer-sized slot in guest
// memory; we store a pointer to a host object in it. A zero slot is a static initialiser,
// so objects are created on first use.
#include "host/hle.h"
#include "host/kernel.h"
#include "host/tls.h"

#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace host {

namespace {

int32_t ret(int hostErr) { return hostErr ? sceErrorFromHost(hostErr) : 0; }

std::mutex gInitMutex; // guards lazy creation of statically initialised objects

// --- threads --------------------------------------------------------------------------

struct Attr {
    uint64_t stackSize = 0x10000;
    int32_t detachState = 0;
    int32_t inheritSched = 4;
    int32_t policy = 1;
    int32_t prio = 700;
    uint64_t affinity = 0x7F;
};

struct Thread {
    pthread_t host{};
    std::string name;
    void* (*entry)(void*) = nullptr;
    void* arg = nullptr;
    void* result = nullptr;
    Attr attr;
    jmp_buf exitJump;
};

thread_local Thread* tSelf = nullptr;

Thread* self() {
    if (!tSelf) {
        tSelf = new Thread;
        tSelf->host = pthread_self();
        tSelf->name = "main";
    }
    return tSelf;
}

void* threadStart(void* p) {
    auto* t = static_cast<Thread*>(p);
    tSelf = t;
    createGuestTls();
    pthread_setname_np(pthread_self(), t->name.substr(0, 15).c_str());
    if (setjmp(t->exitJump) == 0) t->result = t->entry(t->arg);
    return t->result;
}

int32_t scePthreadAttrInit(Attr** attr) {
    *attr = new Attr;
    return 0;
}
int32_t scePthreadAttrDestroy(Attr** attr) {
    delete *attr;
    *attr = nullptr;
    return 0;
}
int32_t scePthreadAttrSetstacksize(Attr** a, uint64_t v) { return (*a)->stackSize = v, 0; }
int32_t scePthreadAttrGetstacksize(Attr** a, uint64_t* v) { return *v = (*a)->stackSize, 0; }
int32_t scePthreadAttrSetdetachstate(Attr** a, int32_t v) { return (*a)->detachState = v, 0; }
int32_t scePthreadAttrGetdetachstate(Attr** a, int32_t* v) { return *v = (*a)->detachState, 0; }
int32_t scePthreadAttrSetinheritsched(Attr** a, int32_t v) { return (*a)->inheritSched = v, 0; }
int32_t scePthreadAttrSetschedpolicy(Attr** a, int32_t v) { return (*a)->policy = v, 0; }
int32_t scePthreadAttrSetschedparam(Attr** a, const int32_t* prio) { return (*a)->prio = *prio, 0; }
int32_t scePthreadAttrGetschedparam(Attr** a, int32_t* prio) { return *prio = (*a)->prio, 0; }
int32_t scePthreadAttrSetaffinity(Attr** a, uint64_t mask) { return (*a)->affinity = mask, 0; }
int32_t scePthreadAttrGetaffinity(Attr** a, uint64_t* mask) { return *mask = (*a)->affinity, 0; }
int32_t scePthreadAttrGet(Thread* t, Attr** a) {
    if (!t || !a || !*a) return sceError(ps4errno::kEINVAL);
    **a = t->attr;
    return 0;
}

int32_t scePthreadCreate(Thread** out, Attr** attr, void* (*entry)(void*), void* arg, const char* name) {
    if (!out || !entry) return sceError(ps4errno::kEINVAL);
    auto* t = new Thread;
    t->entry = entry;
    t->arg = arg;
    t->name = name ? name : "";
    if (attr && *attr) t->attr = **attr;

    pthread_attr_t ha;
    pthread_attr_init(&ha);
    // Host code (HLE, logging) shares the guest's stack, so never go below 1 MiB.
    pthread_attr_setstacksize(&ha, std::max<uint64_t>(t->attr.stackSize, 1 << 20));
    if (t->attr.detachState) pthread_attr_setdetachstate(&ha, PTHREAD_CREATE_DETACHED);
    const int e = pthread_create(&t->host, &ha, threadStart, t);
    pthread_attr_destroy(&ha);
    if (e) {
        delete t;
        return sceErrorFromHost(e);
    }
    std::fprintf(stderr, "[thread] created \"%s\"\n", t->name.c_str());
    *out = t;
    return 0;
}

Thread* scePthreadSelf() { return self(); }
int32_t scePthreadEqual(Thread* a, Thread* b) { return a == b; }
int32_t scePthreadJoin(Thread* t, void** value) {
    WaitLog w{"join", -1, __builtin_frame_address(0)};
    const int e = pthread_join(t->host, nullptr);
    if (e) return sceErrorFromHost(e);
    if (value) *value = t->result;
    return 0;
}
int32_t scePthreadDetach(Thread* t) { return ret(pthread_detach(t->host)); }
void scePthreadExit(void* value) {
    Thread* t = self();
    t->result = value;
    if (t->entry) longjmp(t->exitJump, 1); // unwind guest frames without the host unwinder
    std::fprintf(stderr, "[thread] main thread called scePthreadExit\n");
    pthread_exit(value);
}
int32_t scePthreadYield() { return sched_yield(), 0; }
int32_t scePthreadGetname(Thread* t, char* name) {
    std::snprintf(name, 32, "%s", t->name.c_str());
    return 0;
}
int32_t scePthreadRename(Thread* t, const char* name) {
    t->name = name ? name : "";
    return 0;
}
int32_t scePthreadSetprio(Thread* t, int32_t prio) { return t->attr.prio = prio, 0; }
int32_t scePthreadGetprio(Thread* t, int32_t* prio) { return *prio = t->attr.prio, 0; }
int32_t scePthreadSetaffinity(Thread* t, uint64_t mask) { return t->attr.affinity = mask, 0; }
int32_t scePthreadGetaffinity(Thread* t, uint64_t* mask) { return *mask = t->attr.affinity, 0; }

// --- mutexes --------------------------------------------------------------------------

struct MutexAttr {
    int32_t type = 1; // PS4: 1 errorcheck, 2 recursive, 3 normal, 4 adaptive
    int32_t protocol = 0;
};

struct Mutex {
    pthread_mutex_t m;
    std::string name;
};

Mutex* newMutex(const MutexAttr* a, const char* name) {
    auto* mx = new Mutex;
    pthread_mutexattr_t ha;
    pthread_mutexattr_init(&ha);
    const int32_t type = a ? a->type : 1;
    pthread_mutexattr_settype(&ha, type == 2   ? PTHREAD_MUTEX_RECURSIVE
                                   : type == 1 ? PTHREAD_MUTEX_ERRORCHECK
                                               : PTHREAD_MUTEX_NORMAL);
    pthread_mutex_init(&mx->m, &ha);
    pthread_mutexattr_destroy(&ha);
    mx->name = name ? name : "";
    return mx;
}

Mutex* getMutex(Mutex** slot) {
    if (!slot) return nullptr;
    Mutex* m = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (m && uintptr_t(m) > 16) return m;
    std::lock_guard lock(gInitMutex);
    if (!*slot || uintptr_t(*slot) <= 16) {
        MutexAttr a;
        if (uintptr_t(*slot) == 1) a.type = 4; // PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
        __atomic_store_n(slot, newMutex(&a, nullptr), __ATOMIC_RELEASE);
    }
    return *slot;
}

int32_t scePthreadMutexattrInit(MutexAttr** a) { return *a = new MutexAttr, 0; }
int32_t scePthreadMutexattrDestroy(MutexAttr** a) {
    delete *a;
    *a = nullptr;
    return 0;
}
int32_t scePthreadMutexattrSettype(MutexAttr** a, int32_t type) { return (*a)->type = type, 0; }
int32_t scePthreadMutexattrSetprotocol(MutexAttr** a, int32_t p) { return (*a)->protocol = p, 0; }

int32_t scePthreadMutexInit(Mutex** slot, MutexAttr** attr, const char* name) {
    if (!slot) return sceError(ps4errno::kEINVAL);
    *slot = newMutex(attr && *attr ? *attr : nullptr, name);
    return 0;
}
int32_t scePthreadMutexDestroy(Mutex** slot) {
    if (!slot || !*slot) return sceError(ps4errno::kEINVAL);
    pthread_mutex_destroy(&(*slot)->m);
    delete *slot;
    *slot = nullptr;
    return 0;
}
int32_t scePthreadMutexLock(Mutex** slot) {
    Mutex* m = getMutex(slot);
    if (!m) return sceError(ps4errno::kEINVAL);
    if (pthread_mutex_trylock(&m->m) == 0) return 0;
    WaitLog w{"mutex lock", -1, __builtin_frame_address(0)};
    return ret(pthread_mutex_lock(&m->m));
}
int32_t scePthreadMutexTrylock(Mutex** slot) {
    Mutex* m = getMutex(slot);
    return m ? ret(pthread_mutex_trylock(&m->m)) : sceError(ps4errno::kEINVAL);
}
int32_t scePthreadMutexUnlock(Mutex** slot) {
    Mutex* m = getMutex(slot);
    return m ? ret(pthread_mutex_unlock(&m->m)) : sceError(ps4errno::kEINVAL);
}

// --- condition variables --------------------------------------------------------------

struct Cond {
    pthread_cond_t c;
};

Cond* getCond(Cond** slot) {
    Cond* c = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (c) return c;
    std::lock_guard lock(gInitMutex);
    if (!*slot) {
        auto* n = new Cond;
        pthread_condattr_t ca;
        pthread_condattr_init(&ca);
        pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
        pthread_cond_init(&n->c, &ca);
        pthread_condattr_destroy(&ca);
        __atomic_store_n(slot, n, __ATOMIC_RELEASE);
    }
    return *slot;
}

int32_t scePthreadCondattrInit(void** a) { return *a = new int(0), 0; }
int32_t scePthreadCondattrDestroy(void** a) {
    delete static_cast<int*>(*a);
    *a = nullptr;
    return 0;
}
int32_t scePthreadCondInit(Cond** slot, void**, const char*) {
    *slot = nullptr;
    getCond(slot);
    return 0;
}
int32_t scePthreadCondDestroy(Cond** slot) {
    if (*slot) {
        pthread_cond_destroy(&(*slot)->c);
        delete *slot;
        *slot = nullptr;
    }
    return 0;
}
int32_t scePthreadCondWait(Cond** c, Mutex** m) {
    WaitLog w{"cond wait", -1, __builtin_frame_address(0)};
    return ret(pthread_cond_wait(&getCond(c)->c, &getMutex(m)->m));
}
int32_t scePthreadCondTimedwait(Cond** c, Mutex** m, uint32_t usec) {
    WaitLog w{"cond timedwait", usec, __builtin_frame_address(0)};
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ts.tv_sec += usec / 1000000;
    ts.tv_nsec += long(usec % 1000000) * 1000;
    if (ts.tv_nsec >= 1000000000) ts.tv_sec++, ts.tv_nsec -= 1000000000;
    return ret(pthread_cond_timedwait(&getCond(c)->c, &getMutex(m)->m, &ts));
}
int32_t scePthreadCondSignal(Cond** c) { return ret(pthread_cond_signal(&getCond(c)->c)); }
int32_t scePthreadCondBroadcast(Cond** c) { return ret(pthread_cond_broadcast(&getCond(c)->c)); }

// --- rwlocks --------------------------------------------------------------------------

pthread_rwlock_t* getRwlock(pthread_rwlock_t** slot) {
    pthread_rwlock_t* l = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (l) return l;
    std::lock_guard lock(gInitMutex);
    if (!*slot) {
        auto* n = new pthread_rwlock_t;
        pthread_rwlock_init(n, nullptr);
        __atomic_store_n(slot, n, __ATOMIC_RELEASE);
    }
    return *slot;
}
int32_t scePthreadRwlockInit(pthread_rwlock_t** s, void**, const char*) {
    *s = nullptr;
    getRwlock(s);
    return 0;
}
int32_t scePthreadRwlockDestroy(pthread_rwlock_t** s) {
    if (*s) {
        pthread_rwlock_destroy(*s);
        delete *s;
        *s = nullptr;
    }
    return 0;
}
int32_t scePthreadRwlockRdlock(pthread_rwlock_t** s) { return ret(pthread_rwlock_rdlock(getRwlock(s))); }
int32_t scePthreadRwlockWrlock(pthread_rwlock_t** s) { return ret(pthread_rwlock_wrlock(getRwlock(s))); }
int32_t scePthreadRwlockTryrdlock(pthread_rwlock_t** s) { return ret(pthread_rwlock_tryrdlock(getRwlock(s))); }
int32_t scePthreadRwlockTrywrlock(pthread_rwlock_t** s) { return ret(pthread_rwlock_trywrlock(getRwlock(s))); }
int32_t scePthreadRwlockUnlock(pthread_rwlock_t** s) { return ret(pthread_rwlock_unlock(getRwlock(s))); }

// --- thread-specific data and once ----------------------------------------------------

int32_t scePthreadKeyCreate(int32_t* key, void (*dtor)(void*)) {
    pthread_key_t k;
    const int e = pthread_key_create(&k, dtor);
    if (e) return sceErrorFromHost(e);
    *key = int32_t(k);
    return 0;
}
int32_t scePthreadKeyDelete(int32_t key) { return ret(pthread_key_delete(pthread_key_t(key))); }
void* scePthreadGetspecific(int32_t key) { return pthread_getspecific(pthread_key_t(key)); }
int32_t scePthreadSetspecific(int32_t key, const void* v) { return ret(pthread_setspecific(pthread_key_t(key), v)); }

// FreeBSD pthread_once_t: { int state; pthread_mutex_t mutex; }, state 0 = not run, 1 = done.
int32_t scePthreadOnce(int32_t* once, void (*fn)()) {
    enum { kNeedsInit = 0, kDone = 1, kRunning = 2 };
    int32_t expected = kNeedsInit;
    if (__atomic_compare_exchange_n(once, &expected, kRunning, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        fn();
        __atomic_store_n(once, kDone, __ATOMIC_RELEASE);
        return 0;
    }
    while (__atomic_load_n(once, __ATOMIC_ACQUIRE) != kDone) sched_yield();
    return 0;
}

// --- semaphores -----------------------------------------------------------------------

struct Sema {
    std::mutex m;
    std::condition_variable cv;
    int32_t count, max;
    std::string name;
};

int32_t sceKernelCreateSema(Sema** out, const char* name, uint32_t, int32_t init, int32_t max, const void*) {
    if (!out || init < 0 || max <= 0 || init > max) return sceError(ps4errno::kEINVAL);
    *out = new Sema{{}, {}, init, max, name ? name : ""};
    return 0;
}
int32_t sceKernelDeleteSema(Sema* s) {
    delete s;
    return 0;
}
int32_t sceKernelWaitSema(Sema* s, int32_t need, uint32_t* timeoutUs) {
    if (!s || need <= 0 || need > s->max) return sceError(ps4errno::kEINVAL);
    WaitLog w{"sema wait", timeoutUs ? int64_t(*timeoutUs) : -1, __builtin_frame_address(0)};
    std::unique_lock lock(s->m);
    auto ready = [&] { return s->count >= need; };
    if (timeoutUs) {
        const auto start = std::chrono::steady_clock::now();
        const bool ok = s->cv.wait_for(lock, std::chrono::microseconds(*timeoutUs), ready);
        const auto used = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
        *timeoutUs = ok ? uint32_t(std::max<int64_t>(0, int64_t(*timeoutUs) - used.count())) : 0;
        if (!ok) return sceError(ps4errno::kETIMEDOUT);
    } else {
        s->cv.wait(lock, ready);
    }
    s->count -= need;
    return 0;
}
int32_t sceKernelPollSema(Sema* s, int32_t need) {
    if (!s || need <= 0) return sceError(ps4errno::kEINVAL);
    std::lock_guard lock(s->m);
    if (s->count < need) return sceError(ps4errno::kEBUSY);
    s->count -= need;
    return 0;
}
int32_t sceKernelSignalSema(Sema* s, int32_t n) {
    if (!s) return sceError(ps4errno::kEINVAL);
    {
        std::lock_guard lock(s->m);
        if (s->count + n > s->max) return sceError(ps4errno::kEINVAL);
        s->count += n;
    }
    s->cv.notify_all();
    return 0;
}

} // namespace

void registerPthreadHle() {
    const char* k = "libkernel";
    HLE(k, scePthreadAttrInit);
    HLE(k, scePthreadAttrDestroy);
    HLE(k, scePthreadAttrSetstacksize);
    HLE(k, scePthreadAttrGetstacksize);
    HLE(k, scePthreadAttrSetdetachstate);
    HLE(k, scePthreadAttrGetdetachstate);
    HLE(k, scePthreadAttrSetinheritsched);
    HLE(k, scePthreadAttrSetschedpolicy);
    HLE(k, scePthreadAttrSetschedparam);
    HLE(k, scePthreadAttrGetschedparam);
    HLE(k, scePthreadAttrSetaffinity);
    HLE(k, scePthreadAttrGetaffinity);
    HLE(k, scePthreadAttrGet);
    HLE(k, scePthreadCreate);
    HLE(k, scePthreadSelf);
    HLE(k, scePthreadEqual);
    HLE(k, scePthreadJoin);
    HLE(k, scePthreadDetach);
    HLE(k, scePthreadExit);
    HLE(k, scePthreadYield);
    HLE(k, scePthreadGetname);
    HLE(k, scePthreadRename);
    HLE(k, scePthreadSetprio);
    HLE(k, scePthreadGetprio);
    HLE(k, scePthreadSetaffinity);
    HLE(k, scePthreadGetaffinity);
    HLE(k, scePthreadMutexattrInit);
    HLE(k, scePthreadMutexattrDestroy);
    HLE(k, scePthreadMutexattrSettype);
    HLE(k, scePthreadMutexattrSetprotocol);
    HLE(k, scePthreadMutexInit);
    HLE(k, scePthreadMutexDestroy);
    HLE(k, scePthreadMutexLock);
    HLE(k, scePthreadMutexTrylock);
    HLE(k, scePthreadMutexUnlock);
    HLE(k, scePthreadCondattrInit);
    HLE(k, scePthreadCondattrDestroy);
    HLE(k, scePthreadCondInit);
    HLE(k, scePthreadCondDestroy);
    HLE(k, scePthreadCondWait);
    HLE(k, scePthreadCondTimedwait);
    HLE(k, scePthreadCondSignal);
    HLE(k, scePthreadCondBroadcast);
    HLE(k, scePthreadRwlockInit);
    HLE(k, scePthreadRwlockDestroy);
    HLE(k, scePthreadRwlockRdlock);
    HLE(k, scePthreadRwlockWrlock);
    HLE(k, scePthreadRwlockTryrdlock);
    HLE(k, scePthreadRwlockTrywrlock);
    HLE(k, scePthreadRwlockUnlock);
    HLE(k, scePthreadKeyCreate);
    HLE(k, scePthreadKeyDelete);
    HLE(k, scePthreadGetspecific);
    HLE(k, scePthreadSetspecific);
    HLE(k, scePthreadOnce);
    HLE(k, sceKernelCreateSema);
    HLE(k, sceKernelDeleteSema);
    HLE(k, sceKernelWaitSema);
    HLE(k, sceKernelPollSema);
    HLE(k, sceKernelSignalSema);
}

} // namespace host
