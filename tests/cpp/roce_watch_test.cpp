// patches/0350: roce_watch.h's FailWatch on the CPU, with a thread playing the proxy's poll loop and the main thread
// playing the GPU (failure record, then the failure word) and the NIC (the flag, before or after the failure).
// Built and run by tests/test_roce_watch.py; prints one line per scenario: name state host_at_fail gpu_seen late_ms.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <thread>

#include "roce_watch.h"

using namespace tfroce;

namespace {

constexpr int WORLD = 2, SLOTS = 2, NHCA = 2, STRIDE = 32;     // flag stride in words
alignas(128) uint32_t ctrl[32];
alignas(128) uint32_t flags[WORLD * SLOTS * NHCA * STRIDE];

const volatile uint32_t *flag_at(uint32_t p, uint32_t s, uint32_t h) {
    if (p >= WORLD || s >= SLOTS || h >= NHCA) return nullptr;
    return &flags[((p * SLOTS + s) * NHCA + h) * STRIDE];
}

void set_flag(uint32_t p, uint32_t s, uint32_t h, uint32_t v) {
    __atomic_store_n(const_cast<uint32_t *>(flag_at(p, s, h)), v, __ATOMIC_RELEASE);
}

void sleep_ms(int ms) {
    timespec t = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&t, nullptr);
}

// the kernel's failure record: error words, ERR_SEEN, a fence, then the failure word
void fail(uint32_t seq, uint32_t peer, uint32_t hca, uint32_t seen) {
    ctrl[WATCH_CTRL_ERR_PEER] = peer;
    ctrl[WATCH_CTRL_ERR_HCA] = hca;
    ctrl[WATCH_CTRL_ERR_SEQ] = seq;
    ctrl[WATCH_CTRL_ERR_SEEN] = seen;
    __atomic_store_n(&ctrl[WATCH_CTRL_FAILED], 1u, __ATOMIC_RELEASE);
}

struct Poller {
    FailWatch w;
    std::atomic<int> run{1};
    std::thread t;
    Poller() : t([this] {
        while (run.load(std::memory_order_relaxed)) w.poll(ctrl, flag_at);
    }) {}
    ~Poller() {
        run.store(0);
        t.join();
    }
    uint32_t wait_state(uint32_t want, int ms) {
        for (int i = 0; i < ms * 10; i++) {
            if (w.state.load() >= want) break;
            timespec t = {0, 100000};
            nanosleep(&t, nullptr);
        }
        return w.state.load();
    }
};

void reset() {
    memset(ctrl, 0, sizeof(ctrl));
    memset(flags, 0, sizeof(flags));
}

void report(const char *name, Poller &p) {
    uint64_t o[7];
    p.w.read(o);
    printf("%s state=%llu seq=%llu peer=%llu hca=%llu host_at_fail=%llu gpu_seen=%llu late_ms=%.3f\n", name,
           (unsigned long long)o[0], (unsigned long long)o[1], (unsigned long long)o[2], (unsigned long long)o[3],
           (unsigned long long)o[4], (unsigned long long)o[5], o[6] / 1e6);
    fflush(stdout);
}

}  // namespace

int main() {
    {   // idle: no failure word, nothing recorded however long it polls
        reset();
        Poller p;
        set_flag(1, 0, 0, 310);
        sleep_ms(20);
        report("idle", p);
    }
    {   // not_seen: the flag is in host memory before the failure; the GPU's last read was stale
        reset();
        Poller p;
        set_flag(1, 0, 1, 312);
        sleep_ms(2);
        fail(312, 1, 1, 310);
        p.wait_state(1, 1000);
        sleep_ms(10);
        report("not_seen", p);
    }
    {   // late: the flag comes 30 ms after the failure (the W2 loopback)
        reset();
        Poller p;
        set_flag(1, 0, 0, 310);
        fail(312, 1, 0, 310);
        p.wait_state(1, 1000);
        sleep_ms(30);
        set_flag(1, 0, 0, 312);
        p.wait_state(2, 1000);
        report("late", p);
    }
    {   // never
        reset();
        Poller p;
        set_flag(1, 1, 0, 311);
        fail(313, 1, 0, 311);
        p.wait_state(1, 1000);
        sleep_ms(30);
        report("never", p);
    }
    {   // an out-of-range record (a torn or garbage record must not crash the proxy)
        reset();
        Poller p;
        fail(5, 7, 9, 0);
        p.wait_state(1, 1000);
        report("range", p);
    }
    {   // many records written by a racing "GPU" thread: every one read whole (fields agree with each other)
        int bad = 0;
        for (int i = 0; i < 2000; i++) {
            reset();
            Poller p;
            const uint32_t seq = 1000 + i, peer = i & 1, hca = (i >> 1) & 1;
            std::thread gpu([&] { fail(seq, peer, hca, seq - 2); });
            p.wait_state(1, 1000);
            gpu.join();
            uint64_t o[7];
            p.w.read(o);
            if (o[0] != 1 || o[1] != seq || o[2] != peer || o[3] != hca || o[5] != seq - 2) bad++;
        }
        printf("race bad=%d\n", bad);
    }
    return 0;
}
