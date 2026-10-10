// Card codes and preflop classes — mirrors poker_alpha/poker/cards.py.
//
// A card is an integer 0..51: rank = code % 13 (0 = deuce .. 12 = ace),
// suit = code / 13 (clubs, diamonds, hearts, spades).
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace pa {

constexpr int NUM_CARDS = 52;
constexpr char RANK_CHARS[] = "23456789TJQKA";
constexpr char SUIT_CHARS[] = "cdhs";

inline int rank_of(int c) { return c % 13; }
inline int suit_of(int c) { return c / 13; }

inline std::string card_str(int code) {
    std::string s;
    s += RANK_CHARS[rank_of(code)];
    s += SUIT_CHARS[suit_of(code)];
    return s;
}

// Preflop class id: a bijection {0..168} <-> the 169 starting-hand classes.
// Layout: pairs 0..12 by rank, then for hi > lo:
// id = 13 + 2*pair_index(hi, lo) + (suited ? 0 : 1), with pair_index the
// position of (hi, lo) in the hi-descending, lo-descending enumeration.
// The exact order is irrelevant (only bijectivity and rendering matter).
inline int preflop_class_id(int a, int b) {
    int ra = rank_of(a), rb = rank_of(b);
    int hi = ra > rb ? ra : rb, lo = ra > rb ? rb : ra;
    if (hi == lo) return hi;
    bool suited = suit_of(a) == suit_of(b);
    // index of (hi, lo) among all 78 ordered pairs with hi > lo:
    // pairs with first rank > hi come first? Use simple formula:
    // count of (h,l) pairs with h < hi: hi*(hi-1)/2; within hi, lo ranges
    // 0..hi-1.
    int pair_index = hi * (hi - 1) / 2 + lo;
    return 13 + 2 * pair_index + (suited ? 0 : 1);
}

inline std::string preflop_class_name(int id) {
    std::string s;
    if (id < 13) {
        s += RANK_CHARS[id];
        s += RANK_CHARS[id];
        return s;
    }
    int rest = id - 13;
    bool suited = rest % 2 == 0;
    int pair_index = rest / 2;
    // invert pair_index = hi*(hi-1)/2 + lo
    int hi = 1;
    while ((hi + 1) * hi / 2 <= pair_index) ++hi;
    int lo = pair_index - hi * (hi - 1) / 2;
    s += RANK_CHARS[hi];
    s += RANK_CHARS[lo];
    s += suited ? 's' : 'o';
    return s;
}

}  // namespace pa
