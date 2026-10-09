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

// Best 5-card value from 5-7 card codes. Caller passes valid distinct codes.
uint32_t evaluate(const int* cards, int n);

inline int eval_category(uint32_t v) { return int(v >> 20); }

}  // namespace pa
