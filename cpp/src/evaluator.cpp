#include "evaluator.hpp"

namespace pa {

namespace {

constexpr uint32_t ACE_BIT = 1u << 12;
constexpr uint32_t WHEEL_LOW = 0b1111u;  // deuce..five

// High rank of the best straight in a 13-bit rank mask, or -1.
inline int straight_high(uint32_t rank_mask) {
    uint32_t m = rank_mask & (rank_mask >> 1) & (rank_mask >> 2) &
                 (rank_mask >> 3) & (rank_mask >> 4);
    if (m) {
        int bl = 0;
        while (m >> bl) ++bl;        // bit_length
        return bl + 3;               // (bit_length - 1) + 4
    }
    if ((rank_mask & ACE_BIT) && (rank_mask & WHEEL_LOW) == WHEEL_LOW)
        return 3;                    // wheel: straight to the five
    return -1;
}

inline uint32_t pack1(int cat, int a) { return uint32_t(cat) << 20 | uint32_t(a) << 16; }
inline uint32_t pack2(int cat, int a, int b) {
    return pack1(cat, a) | uint32_t(b) << 12;
}
inline uint32_t pack3(int cat, int a, int b, int c) {
    return pack2(cat, a, b) | uint32_t(c) << 8;
}
inline uint32_t pack4(int cat, int a, int b, int c, int d) {
    return pack3(cat, a, b, c) | uint32_t(d) << 4;
}
inline uint32_t pack5(int cat, int a, int b, int c, int d, int e) {
    return pack4(cat, a, b, c, d) | uint32_t(e);
}

}  // namespace

uint32_t evaluate(const int* cards, int n) {
    int rank_count[13] = {};
    int suit_count[4] = {};
    uint32_t suit_mask[4] = {};
    uint32_t rank_mask = 0;
    for (int i = 0; i < n; ++i) {
        int c = cards[i];
        int r = c % 13, s = c / 13;
        rank_count[r] += 1;
        suit_count[s] += 1;
        suit_mask[s] |= 1u << r;
        rank_mask |= 1u << r;
    }

    int flush_suit = -1;
    for (int s = 0; s < 4; ++s)
        if (suit_count[s] >= 5) { flush_suit = s; break; }

    if (flush_suit >= 0) {
        int high = straight_high(suit_mask[flush_suit]);
        if (high >= 0) return pack1(STRAIGHT_FLUSH, high);
    }

    // Ranks present, highest first, split by multiplicity.
    int present[7], np = 0;
    int quads[2], nq = 0;
    int trips[2], nt = 0;
    int pairs[3], npair = 0;
    for (int r = 12; r >= 0; --r) {
        if (!rank_count[r]) continue;
        present[np++] = r;
        if (rank_count[r] == 4) quads[nq++] = r;
        else if (rank_count[r] == 3) trips[nt++] = r;
        else if (rank_count[r] == 2) pairs[npair++] = r;
    }

    if (nq) {
        int q = quads[0];
        for (int i = 0; i < np; ++i)
            if (present[i] != q) return pack2(FOUR_OF_A_KIND, q, present[i]);
    }

    if (nt) {
        int t = trips[0];
        // The boat's pair may be a second set of trips or the best pair.
        int best = -1;
        if (nt > 1) best = trips[1];
        if (npair && pairs[0] > best) best = pairs[0];
        if (best >= 0) return pack2(FULL_HOUSE, t, best);
    }

    if (flush_suit >= 0) {
        uint32_t fm = suit_mask[flush_suit];
        int t[5], k = 0;
        for (int r = 12; r >= 0 && k < 5; --r)
            if ((fm >> r) & 1u) t[k++] = r;
        return pack5(FLUSH, t[0], t[1], t[2], t[3], t[4]);
    }

    {
        int high = straight_high(rank_mask);
        if (high >= 0) return pack1(STRAIGHT, high);
    }

    if (nt) {
        int t = trips[0];
        int k[2], nk = 0;
        for (int i = 0; i < np && nk < 2; ++i)
            if (present[i] != t) k[nk++] = present[i];
        return pack3(THREE_OF_A_KIND, t, k[0], k[1]);
    }

    if (npair >= 2) {
        int hi = pairs[0], lo = pairs[1];
        for (int i = 0; i < np; ++i)
            if (present[i] != hi && present[i] != lo)
                return pack3(TWO_PAIR, hi, lo, present[i]);
    }

    if (npair == 1) {
        int p = pairs[0];
        int k[3], nk = 0;
        for (int i = 0; i < np && nk < 3; ++i)
            if (present[i] != p) k[nk++] = present[i];
        return pack4(ONE_PAIR, p, k[0], k[1], k[2]);
    }

    return pack5(HIGH_CARD, present[0], present[1], present[2], present[3],
                 present[4]);
}

}  // namespace pa
