// Randomness for the native solver.
//
// Production RNG: xoshiro256** (Blackman & Vigna, public domain reference
// algorithm), seeded from the user seed via splitmix64. State = 4 x uint64,
// serialized into checkpoints as "xoshiro256**/v1".
//
// Random tape: a testing-only source that replays a pre-generated sequence
// of uniform doubles. The same tape is consumable by the Python reference
// solver (tests/native/tape.py), giving exact cross-backend trajectory
// parity without reimplementing NumPy's generators.
//
// Cross-backend bit identity of the *production* streams is explicitly not
// claimed: NumPy PCG64 and xoshiro256** differ by design.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pa {

struct Xoshiro256 {
    uint64_t s[4];

    static uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    explicit Xoshiro256(uint64_t seed = 0) {
        uint64_t x = seed;
        for (auto& w : s) w = splitmix64(x);
    }

    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    uint64_t next_u64() {
        uint64_t result = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Uniform double in [0, 1), 53 bits.
    double next_double() { return (next_u64() >> 11) * 0x1.0p-53; }

    // Unbiased integer in [0, n) via rejection sampling (deterministic
    // given the stream).
    uint32_t next_below(uint32_t n) {
        uint64_t threshold = (~uint64_t(0)) - (~uint64_t(0)) % n;
        for (;;) {
            uint64_t r = next_u64();
            if (r < threshold) return uint32_t(r % n);
        }
    }
};

// Source abstraction: production RNG or test tape.
class RandomSource {
public:
    explicit RandomSource(uint64_t seed) : rng_(seed), tape_mode_(false) {}
    explicit RandomSource(std::vector<double> tape)
        : rng_(0), tape_(std::move(tape)), tape_mode_(true) {}

    bool tape_mode() const { return tape_mode_; }
    std::size_t tape_pos() const { return tape_pos_; }

    double uniform() {
        if (!tape_mode_) return rng_.next_double();
        if (tape_pos_ >= tape_.size())
            throw std::runtime_error("random tape exhausted");
        return tape_[tape_pos_++];
    }

    // Integer in [0, n). Tape mode uses floor(u * n) — the exact rule the
    // Python tape helper uses — so trajectories match; production mode uses
    // unbiased rejection sampling.
    int below(int n) {
        if (tape_mode_) {
            int i = int(uniform() * n);
            return i >= n ? n - 1 : i;  // guard u == 1.0-epsilon edge
        }
        return int(rng_.next_below(uint32_t(n)));
    }

    // Serialize / restore production state (tape mode is never checkpointed).
    void get_state(uint64_t out[4]) const {
        for (int i = 0; i < 4; ++i) out[i] = rng_.s[i];
    }
    void set_state(const uint64_t in[4]) {
        for (int i = 0; i < 4; ++i) rng_.s[i] = in[i];
    }

private:
    Xoshiro256 rng_;
    std::vector<double> tape_;
    std::size_t tape_pos_ = 0;
    bool tape_mode_;
};

}  // namespace pa
