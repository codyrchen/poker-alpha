// Hold'em rules — native port of poker_alpha/games/holdem.py (HoldemGame),
// maintained incrementally on HoldemState (see state.hpp). Every chip
// amount is computed with the identical IEEE expressions in the identical
// order as the Python replay, so doubles are bit-equal.
#pragma once

#include <cstdint>

#include "config.hpp"
#include "rng.hpp"
#include "state.hpp"

namespace pa {

constexpr int BOARD_SIZE_AT[4] = {0, 3, 4, 5};  // per street

class HoldemGame {
public:
    explicit HoldemGame(const NativeConfig& cfg) : cfg_(cfg) { cfg_.validate(); }

    const NativeConfig& config() const { return cfg_; }

    HoldemState root() const;

    bool is_chance(const HoldemState& s) const;
    bool is_terminal(const HoldemState& s) const;
    bool betting_closed(const HoldemState& s) const;
    int current_player(const HoldemState& s) const { return s.to_act; }

    // Utility for player 0 at a terminal state.
    double utility(const HoldemState& s) const;

    // Legal action token ids at a decision node, in Python order
    // ([f?] c sized... a). Returns the count; also usable as a 9-bit mask
    // via legal_mask().
    int legal_actions(const HoldemState& s, int out[MAX_ACTIONS]) const;
    uint16_t legal_mask(const HoldemState& s) const;

    // Apply a decision action (token id); returns the successor by value.
    HoldemState next_state(const HoldemState& s, int tok) const;

    // Sample one chance successor (root deal / street deal / runout).
    HoldemState sample_chance(const HoldemState& s, RandomSource& rnd) const;

    // Chips added by sized token `tok` (HoldemGame.raise_add).
    double raise_add(int tok, double owe, double pot_now, double my_street_paid) const;

    // Debug/test construction: install specific hole cards at the root, or
    // deal specific board cards for the next street (parity tests replay
    // Python trajectories through these instead of sampling).
    HoldemState with_holes(const HoldemState& s, const uint8_t holes[4]) const;
    HoldemState deal_street(const HoldemState& s, const uint8_t* cards, int n) const;

private:
    void start_street(HoldemState& s) const;  // street-transition fold
    NativeConfig cfg_;
};

}  // namespace pa
