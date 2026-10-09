// Native Hold'em state — the incremental equivalent of HoldemState plus the
// betting-replay fold (poker_alpha/games/holdem.py::_replay_uncached) and
// the betting-context fold (poker_alpha/abstraction/betting_history.py).
//
// Python derives street_paid / totals / to_act / raise count / min raise /
// betting context by replaying the action-token history from the blinds on
// every query. The replay is a left fold over tokens; this struct keeps the
// fold state up to date incrementally with the *identical arithmetic in the
// identical order*, so every double is bit-equal to Python's. A compact
// token history is retained for debugging / canonical rendering only.
#pragma once

#include <array>
#include <cstdint>

#include "config.hpp"

namespace pa {

// Betting-context classes (betting_history.py). The key renders only the
// first letter of own_prior, which merges CHECK and CALL ('c'); FACING names
// are rendered in full.
enum class Facing : uint8_t { NONE = 0, SMALL, MEDIUM, LARGE, OVER, ALLIN };
enum class Prior : uint8_t { NONE = 0, CHECK, CALL, AGGR, FOLD };

inline const char* facing_name(Facing f) {
    switch (f) {
        case Facing::NONE: return "none";
        case Facing::SMALL: return "small";
        case Facing::MEDIUM: return "medium";
        case Facing::LARGE: return "large";
        case Facing::OVER: return "over";
        case Facing::ALLIN: return "allin";
    }
    return "?";
}

inline char prior_char(Prior p) {
    switch (p) {
        case Prior::NONE: return 'n';
        case Prior::CHECK: return 'c';
        case Prior::CALL: return 'c';
        case Prior::AGGR: return 'a';
        case Prior::FOLD: return 'f';
    }
    return '?';
}

struct HoldemState {
    // Cards. board_n == 0 and !dealt => root chance node.
    std::array<std::array<uint8_t, 2>, 2> holes{};
    std::array<uint8_t, 5> board{};
    uint8_t board_n = 0;
    bool dealt = false;

    // Streets. street == number of street entries - 1 in Python terms.
    uint8_t street = 0;          // 0 preflop .. 3 river
    uint8_t to_act = 0;
    int8_t folded = -1;
    bool all_in = false;

    // Betting fold state (bit-equal to Python replay).
    std::array<double, 2> total{SMALL_BLIND, BIG_BLIND};
    std::array<double, 2> street_paid{SMALL_BLIND, BIG_BLIND};
    uint8_t n_raises = 0;        // aggressive actions this street
    double min_inc = BIG_BLIND;  // NLHE minimum raise increment this street
    uint8_t tokens_this_street = 0;
    uint8_t last_token = 255;    // token id of the last action this street

    // Betting-context fold state (betting_history.py semantics).
    std::array<Facing, 2> facing{Facing::NONE, Facing::NONE};
    std::array<Prior, 2> own_prior{Prior::NONE, Prior::NONE};
    int8_t street_last_aggr = -1;  // aggressor on this street, or -1
    int8_t last_aggr_prev = -1;    // aggressor on the most recent earlier street with one
    uint8_t spr_bucket = 0;        // bucketized eff/pot at street start

    // Debug/trace history: token ids, with per-street counts.
    std::array<uint8_t, 28> hist{};
    uint8_t hist_n = 0;
    std::array<uint8_t, 4> street_tokens{};  // tokens per street

    double pot() const { return total[0] + total[1]; }
};

}  // namespace pa
