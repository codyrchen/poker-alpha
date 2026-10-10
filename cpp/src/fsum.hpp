// Exactly rounded floating-point summation, equivalent to Python's
// math.fsum (Shewchuk's expansion algorithm with correct final rounding).
//
// The Python solver computes "strategy . child_values" with math.fsum of
// the elementwise products (poker_alpha/solvers/cfr.py::strategy_dot) to be
// platform-independent; this reproduces it bit-for-bit so random-tape
// trajectories match across backends.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace pa {

// Exactly rounded sum of n doubles (n small; MAX_TERMS covers the solver's
// max action count). Algorithm: maintain a list of non-overlapping partials
// (Shewchuk); finish with the same round-half-even correction CPython uses.
template <std::size_t MAX_TERMS = 8>
inline double fsum(const double* x, std::size_t n) {
    std::array<double, MAX_TERMS + 2> p{};
    std::size_t np = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double v = x[i];
        std::size_t out = 0;
        for (std::size_t j = 0; j < np; ++j) {
            double q = p[j];
            if (std::fabs(v) < std::fabs(q)) { double t = v; v = q; q = t; }
            double hi = v + q;
            double lo = q - (hi - v);
            if (lo != 0.0) p[out++] = lo;
            v = hi;
        }
        p[out++] = v;
        np = out;
    }
    // Sum partials from smallest to largest with CPython's correction step.
    if (np == 0) return 0.0;
    std::size_t i = np;
    double hi = p[--i];
    double lo = 0.0;
    while (i > 0) {
        double v = hi;
        double y = p[--i];
        hi = v + y;
        double yr = hi - v;
        lo = y - yr;
        if (lo != 0.0) break;
    }
    // Round-half-even: if the remainder would round hi away, and the next
    // partial has the same sign as lo, adjust by one ulp (CPython fsum).
    if (i > 0 && ((lo < 0.0 && p[i - 1] < 0.0) || (lo > 0.0 && p[i - 1] > 0.0))) {
        double y = lo * 2.0;
        double v = hi + y;
        double yr = v - hi;
        if (y == yr) hi = v;
    }
    return hi;
}

inline double strategy_dot(const double* strategy, const double* values, std::size_t n) {
    std::array<double, 8> prod{};
    for (std::size_t i = 0; i < n; ++i) prod[i] = strategy[i] * values[i];
    return fsum<8>(prod.data(), n);
}

}  // namespace pa
