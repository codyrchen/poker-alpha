// Card features — native port of poker_alpha/abstraction/features.py
// (CARD_FEATURES_VERSION = 1 semantics), restricted to what the compact v2
// encoder key needs. Pure functions of (hole, board); cached.
#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace pa {

struct CardFeatures {
    uint8_t strength;   // 0..7 made-hand ladder (undefined preflop)
    uint8_t draw;       // 0..3
    uint8_t nut;        // 0..2
    uint8_t blocker;    // 0..2
    uint8_t texture;    // index 0..11: paired*6 + suit*2 + connected
    uint8_t pct_bucket; // river percentile bucket (0 if not computed)
};

// Texture index helpers (texture_code v1: [up][rtf][cd]).
inline char texture_pair_char(int idx) { return (idx / 6) ? 'p' : 'u'; }
inline char texture_suit_char(int idx) { return "rtf"[(idx % 6) / 2]; }
inline char texture_conn_char(int idx) { return (idx % 2) ? 'c' : 'd'; }

class FeatureCache {
public:
    explicit FeatureCache(int river_pct_buckets)
        : river_pct_buckets_(river_pct_buckets) {}

    // hole: 2 cards; board: 3..5 cards (postflop only; preflop uses the
    // class id directly). Cards need not be sorted.
    const CardFeatures& get(const uint8_t hole[2], const uint8_t* board, int board_n);

    size_t feature_entries() const { return cache_.size(); }
    size_t river_board_entries() const { return river_values_.size(); }
    void clear() { cache_.clear(); river_values_.clear(); }

private:
    CardFeatures compute(const int hole[2], const int* board, int board_n);
    const std::vector<uint32_t>& river_values(const int* board_sorted);

    int river_pct_buckets_;
    std::unordered_map<uint64_t, CardFeatures> cache_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> river_values_;
    static constexpr size_t CACHE_LIMIT = 1u << 20;         // mirror lru_cache(1M)
    static constexpr size_t RIVER_CACHE_LIMIT = 4096;       // mirror lru_cache(4096)
};

}  // namespace pa
