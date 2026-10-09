// Compact-v2 infoset key — packed uint64 representation of exactly the
// information partition of CompactHoldemEncoder (history="abstract"), and
// its rendering back to the canonical Python key string.
//
// Layout (38 bits used):
//   bits 0-1   street
//   bit  2     player
//   card part (union, disambiguated by street):
//     preflop: bits 3-11 class id (0..168)
//     postflop: bits 3-7 strength rung or river pct bucket (0..19)
//               bits 8-9 draw, 10-11 nut,
//               12-13 blocker (0 unless river and river_blockers),
//               14-17 texture index (0..11)
//   bits 18-19 initiative (0 none, 1 own, 2 opp)
//   bits 20-21 raises this street (0..3)
//   bits 22-24 facing (Facing enum)
//   bits 25-26 own_prior rendered class (0 n, 1 c, 2 a, 3 f)
//   bits 27-28 spr bucket
//   bits 29-37 legal-action mask (token ids 0..8)
//
// The packing is a bijection between field tuples and keys, and the field
// tuple is a bijection with the canonical key string (given the config), so
// native keys collide exactly when Python keys are equal. Parity tests pin
// render(pack(state)) == python infoset_key(state) over large corpora.
#pragma once

#include <cstdint>
#include <string>

#include "config.hpp"
#include "features.hpp"
#include "game.hpp"
#include "state.hpp"

namespace pa {

uint64_t pack_key(const HoldemGame& game, FeatureCache& features,
                  const HoldemState& s);

std::string render_key(const NativeConfig& cfg, uint64_t key);

}  // namespace pa
