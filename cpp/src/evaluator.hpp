// Hand evaluator — native port of poker_alpha/poker/evaluator.py.
//
// evaluate_best_codes returns a Python tuple (category, tiebreakers...)
// compared lexicographically. Within one category every tuple has the same
// length, so packing category and up to five 4-bit tiebreakers into a
// uint32 preserves both the ordering and equality exactly:
//
//     value = cat << 20 | t1 << 16 | t2 << 12 | t3 << 8 | t4 << 4 | t5
//
// (missing tiebreakers are zero). Parity with Python is pinned exhaustively
// over all 2,598,960 five-card hands and over large random 6/7-card corpora
// (tests/native/test_native_evaluator.py).
#pragma once

#include <cstdint>

namespace pa {

constexpr int STRAIGHT_FLUSH = 8, FOUR_OF_A_KIND = 7, FULL_HOUSE = 6,
              FLUSH = 5, STRAIGHT = 4, THREE_OF_A_KIND = 3, TWO_PAIR = 2,
              ONE_PAIR = 1, HIGH_CARD = 0;

// Rank / suit histograms of a partial hand (e.g. a fixed board), so many
// holdings can be evaluated against it without re-counting the shared cards.
struct EvalCounts {
    int rank_count[13] = {};
    int suit_count[4] = {};
    uint32_t suit_mask[4] = {};
    uint32_t rank_mask = 0;
    inline void add(int c) {
        int r = c % 13, s = c / 13;
        rank_count[r] += 1;
        suit_count[s] += 1;
        suit_mask[s] |= 1u << r;
        rank_mask |= 1u << r;
    }
};

// Best-5 value from accumulated counts of 5-7 distinct cards.
uint32_t evaluate_counts(const EvalCounts& h);

// Best 5-card value from 5-7 card codes. Caller passes valid distinct codes.
uint32_t evaluate(const int* cards, int n);

// evaluate(board + {a, b}) for a precomputed board histogram (identical value).
inline uint32_t evaluate_with(const EvalCounts& board, int a, int b) {
    EvalCounts h = board;
    h.add(a);
    h.add(b);
    return evaluate_counts(h);
}

inline int eval_category(uint32_t v) { return int(v >> 20); }

}  // namespace pa
