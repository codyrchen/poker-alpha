#include "features.hpp"

#include <algorithm>

#include "cards.hpp"
#include "evaluator.hpp"

namespace pa {

namespace {

// --- made-hand strength ladder (features.py MADE_STRENGTH) ---------------
enum : uint8_t {
    S_AIR = 0, S_HIGH = 1, S_WEAK_PAIR = 2, S_MID_PAIR = 3, S_TOP_PAIR = 4,
    S_SET_STRAIGHT = 5, S_FLUSH_BOAT = 6, S_QUADS_SF = 7
};

inline bool has_straight(uint32_t rank_mask) {
    uint32_t full = (rank_mask << 1) | ((rank_mask >> 12) & 1u);
    return (full & (full >> 1) & (full >> 2) & (full >> 3) & (full >> 4)) != 0;
}

// made_hand_class -> strength rung (names collapsed; the key only needs the
// rung postflop).
uint8_t made_strength(const int hole[2], const int* board, int board_n) {
    int cards[7];
    cards[0] = hole[0];
    cards[1] = hole[1];
    for (int i = 0; i < board_n; ++i) cards[2 + i] = board[i];
    uint32_t val = evaluate(cards, 2 + board_n);
    int cat = eval_category(val);

    if (board_n == 5 && val == evaluate(board, 5)) return S_AIR;  // board_plays

    int hr0 = rank_of(hole[0]), hr1 = rank_of(hole[1]);
    if (hr0 < hr1) std::swap(hr0, hr1);  // descending
    int bcount[13] = {};
    for (int i = 0; i < board_n; ++i) bcount[rank_of(board[i])] += 1;
    int branks[5], nb = 0;
    for (int r = 12; r >= 0; --r)
        if (bcount[r]) branks[nb++] = r;
    int top = branks[0];
    int max_bcount = 0;
    for (int r = 0; r < 13; ++r) max_bcount = std::max(max_bcount, bcount[r]);
    bool hole_rank_on_board = bcount[hr0] > 0 || bcount[hr1] > 0;

    if (cat == STRAIGHT_FLUSH) return S_QUADS_SF;
    if (cat == FOUR_OF_A_KIND) {
        if (max_bcount == 4 && !hole_rank_on_board) return S_AIR;  // board_plays
        return S_QUADS_SF;
    }
    if (cat == FULL_HOUSE) return S_FLUSH_BOAT;
    if (cat == FLUSH) return S_FLUSH_BOAT;
    if (cat == STRAIGHT) return S_SET_STRAIGHT;
    if (cat == THREE_OF_A_KIND) {
        if (hr0 == hr1 && bcount[hr0] == 1) return S_SET_STRAIGHT;   // set
        if (max_bcount == 3 && !hole_rank_on_board) return S_HIGH;   // board_trips
        return S_SET_STRAIGHT;                                        // trips
    }

    bool pocket = hr0 == hr1;
    int paired_with_board[2], npaired = 0;
    if (bcount[hr0]) paired_with_board[npaired++] = hr0;
    if (!pocket && bcount[hr1]) paired_with_board[npaired++] = hr1;

    if (cat == TWO_PAIR && !pocket && npaired == 2) return S_TOP_PAIR;  // two_pair
    if (pocket) {
        if (hr0 > top) return S_TOP_PAIR;         // overpair
        if (hr0 < branks[nb - 1]) return S_HIGH;  // underpair
        return S_WEAK_PAIR;                       // pocket_middle
    }
    if (npaired) {
        int r = std::max(paired_with_board[0],
                         npaired > 1 ? paired_with_board[1] : -1);
        if (r == top) {
            int kicker = (hr0 == r) ? hr1 : hr0;
            return kicker >= 9 ? S_TOP_PAIR : S_MID_PAIR;  // top pair good/weak
        }
        if (nb > 1 && r == branks[1]) return S_MID_PAIR;   // middle_pair
        return S_WEAK_PAIR;                                 // bottom_pair
    }
    if (hr1 > top) return S_HIGH;        // overcards
    if (hr0 == 12) return S_HIGH;        // ace_high
    if (hr0 >= 10) return S_HIGH;        // king/queen high
    return S_AIR;
}

// draw_type -> DRAW_CLASS (0 none, 1 weak, 2 strong, 3 combo).
uint8_t draw_class(const int hole[2], const int* board, int board_n) {
    if (board_n < 3 || board_n >= 5) return 0;
    int cards[7];
    cards[0] = hole[0];
    cards[1] = hole[1];
    for (int i = 0; i < board_n; ++i) cards[2 + i] = board[i];
    if (eval_category(evaluate(cards, 2 + board_n)) >= STRAIGHT) return 0;

    int suit_total[4] = {}, suit_hole[4] = {};
    for (int i = 0; i < 2 + board_n; ++i) suit_total[suit_of(cards[i])] += 1;
    for (int i = 0; i < 2; ++i) suit_hole[suit_of(hole[i])] += 1;
    bool flush_draw = false, backdoor = false;
    for (int s = 0; s < 4; ++s) {
        if (suit_total[s] == 4 && suit_hole[s]) flush_draw = true;
        if (board_n == 3 && suit_total[s] == 3 && suit_hole[s]) backdoor = true;
    }

    uint32_t hr = (1u << rank_of(hole[0])) | (1u << rank_of(hole[1]));
    uint32_t br = 0;
    for (int i = 0; i < board_n; ++i) br |= 1u << rank_of(board[i]);
    int outs = 0;
    if (!has_straight(hr | br)) {
        for (int x = 0; x < 13; ++x) {
            uint32_t xb = 1u << x;
            if (has_straight(hr | br | xb) && !has_straight(br | xb)) ++outs;
        }
    }
    int straight = outs >= 2 ? 2 : (outs ? 1 : 0);  // open_ended / gutshot / none

    if (flush_draw && straight) return 3;  // combo_draw
    if (flush_draw) return 2;              // flush_draw
    if (straight == 2) return 2;           // open_ended
    if (straight == 1) return 1;           // gutshot
    if (backdoor) return 1;                // backdoor_flush
    return 0;
}

// Exact "no holding beats hero right now" check (features._is_current_nuts).
bool is_current_nuts(const int hole[2], const int* board, int board_n) {
    int cards[7];
    cards[0] = hole[0];
    cards[1] = hole[1];
    for (int i = 0; i < board_n; ++i) cards[2 + i] = board[i];
    uint32_t hv = evaluate(cards, 2 + board_n);
    bool dead[NUM_CARDS] = {};
    for (int i = 0; i < 2 + board_n; ++i) dead[cards[i]] = true;
    int live[NUM_CARDS], nl = 0;
    for (int c = 0; c < NUM_CARDS; ++c)
        if (!dead[c]) live[nl++] = c;
    int opp[7];
    for (int i = 0; i < board_n; ++i) opp[2 + i] = board[i];
    for (int i = 0; i < nl; ++i) {
        for (int j = i + 1; j < nl; ++j) {
            opp[0] = live[i];
            opp[1] = live[j];
            if (evaluate(opp, 2 + board_n) > hv) return false;
        }
    }
    return true;
}

// _flush_blockers: (holds a card of the board's most frequent suit, holds
// the highest missing card of it), for ms >= 2 and board >= 3.
uint8_t flush_blocker(const int hole[2], const int* board, int board_n) {
    int suit_counts[4] = {};
    for (int i = 0; i < board_n; ++i) suit_counts[suit_of(board[i])] += 1;
    int ms = 0;
    for (int s = 0; s < 4; ++s) ms = std::max(ms, suit_counts[s]);
    bool fb = false, nfb = false;
    if (ms >= 2 && board_n >= 3) {
        for (int s = 0; s < 4; ++s) {
            if (suit_counts[s] != ms) continue;
            uint32_t mine = 0;
            for (int i = 0; i < 2; ++i)
                if (suit_of(hole[i]) == s) mine |= 1u << rank_of(hole[i]);
            if (!mine) continue;
            fb = true;
            uint32_t on_board = 0;
            for (int i = 0; i < board_n; ++i)
                if (suit_of(board[i]) == s) on_board |= 1u << rank_of(board[i]);
            int top_missing = -1;
            for (int r = 12; r >= 0; --r)
                if (!((on_board >> r) & 1u)) { top_missing = r; break; }
            if (top_missing >= 0 && ((mine >> top_missing) & 1u)) nfb = true;
        }
    }
    return nfb ? 2 : (fb ? 1 : 0);
}

// texture_code v1 as an index: paired*6 + suit*2 + connected.
uint8_t texture_index(const int* board, int board_n) {
    int rcount[13] = {}, scount[4] = {};
    for (int i = 0; i < board_n; ++i) {
        rcount[rank_of(board[i])] += 1;
        scount[suit_of(board[i])] += 1;
    }
    int paired = 0;
    for (int r = 0; r < 13; ++r)
        if (rcount[r] >= 2) paired = 1;
    int ms = 0;
    for (int s = 0; s < 4; ++s) ms = std::max(ms, scount[s]);
    int suit;
    if (ms >= 3) suit = 2;                        // 'f'
    else if (ms == 2 && board_n < 5) suit = 1;    // 't'
    else suit = 0;                                 // 'r'
    // Connectedness: max distinct ranks of ext inside any 5-rank window,
    // where ext = {r+1 for ranks} | {0 if ace present} (cards.py).
    bool ext[14] = {};
    for (int r = 0; r < 13; ++r)
        if (rcount[r]) ext[r + 1] = true;
    if (rcount[12]) ext[0] = true;
    int conn = 0;
    for (int lo = 0; lo < 10; ++lo) {
        int k = 0;
        for (int r = lo; r < lo + 5; ++r)
            if (ext[r]) ++k;
        conn = std::max(conn, k);
    }
    int connected = (conn >= 3 && board_n >= 3) ? 1 : 0;  // straight_possible
    return uint8_t(paired * 6 + suit * 2 + connected);
}

inline uint64_t pack_cards_key(const int hole_sorted[2], const int* board_sorted,
                               int board_n) {
    uint64_t k = uint64_t(hole_sorted[0]) | (uint64_t(hole_sorted[1]) << 6);
    for (int i = 0; i < board_n; ++i)
        k |= uint64_t(board_sorted[i] + 1) << (12 + 6 * i);  // +1: 0 = absent
    return k;
}

}  // namespace

const std::vector<uint32_t>& FeatureCache::river_values(const int* board_sorted) {
    uint64_t bkey = 0;
    for (int i = 0; i < 5; ++i) bkey |= uint64_t(board_sorted[i]) << (6 * i);
    auto it = river_values_.find(bkey);
    if (it != river_values_.end()) return it->second;
    if (river_values_.size() >= RIVER_CACHE_LIMIT) river_values_.clear();

    // All 1,081 two-card holdings from the 47 non-board cards (hero's own
    // cards are NOT removed — features.py semantics). Rank-pair dedup for
    // holdings that cannot make a flush.
    int suit_n[4] = {};
    bool on_board[NUM_CARDS] = {};
    for (int i = 0; i < 5; ++i) {
        suit_n[suit_of(board_sorted[i])] += 1;
        on_board[board_sorted[i]] = true;
    }
    int live[NUM_CARDS], nl = 0;
    for (int c = 0; c < NUM_CARDS; ++c)
        if (!on_board[c]) live[nl++] = c;
    uint32_t by_ranks[13][13];
    bool have[13][13] = {};
    std::vector<uint32_t> vals;
    vals.reserve(size_t(nl) * (nl - 1) / 2);
    int cards[7];
    for (int i = 0; i < 5; ++i) cards[2 + i] = board_sorted[i];
    for (int i = 0; i < nl; ++i) {
        for (int j = i + 1; j < nl; ++j) {
            int x = live[i], y = live[j];
            if (suit_n[suit_of(x)] >= 3 || suit_n[suit_of(y)] >= 3) {
                cards[0] = x;
                cards[1] = y;
                vals.push_back(evaluate(cards, 7));
                continue;
            }
            int ra = rank_of(x), rb = rank_of(y);
            if (ra > rb) std::swap(ra, rb);
            if (!have[ra][rb]) {
                cards[0] = x;
                cards[1] = y;
                by_ranks[ra][rb] = evaluate(cards, 7);
                have[ra][rb] = true;
            }
            vals.push_back(by_ranks[ra][rb]);
        }
    }
    std::sort(vals.begin(), vals.end());
    return river_values_.emplace(bkey, std::move(vals)).first->second;
}

CardFeatures FeatureCache::compute(const int hole[2], const int* board, int board_n) {
    CardFeatures f{};
    f.strength = made_strength(hole, board, board_n);
    int street = board_n == 3 ? 1 : (board_n == 4 ? 2 : 3);
    f.draw = (street < 3 && f.strength < 5) ? draw_class(hole, board, board_n) : 0;
    if (f.strength >= 5)
        f.nut = is_current_nuts(hole, board, board_n) ? 2 : 1;
    else
        f.nut = 0;
    f.blocker = flush_blocker(hole, board, board_n);
    f.texture = texture_index(board, board_n);
    f.pct_bucket = 0;
    if (river_pct_buckets_ > 0 && board_n == 5) {
        const auto& vals = river_values(board);
        int cards[7];
        cards[0] = hole[0];
        cards[1] = hole[1];
        for (int i = 0; i < 5; ++i) cards[2 + i] = board[i];
        uint32_t hv = evaluate(cards, 7);
        auto lo = std::lower_bound(vals.begin(), vals.end(), hv) - vals.begin();
        auto hi = std::upper_bound(vals.begin(), vals.end(), hv) - vals.begin();
        double pct = (double(lo) + 0.5 * double(hi - lo)) / double(vals.size());
        int b = int(pct * river_pct_buckets_);
        if (b > river_pct_buckets_ - 1) b = river_pct_buckets_ - 1;
        f.pct_bucket = uint8_t(b);
    }
    return f;
}

const CardFeatures& FeatureCache::get(const uint8_t hole[2], const uint8_t* board,
                                      int board_n) {
    int h[2] = {hole[0], hole[1]};
    if (h[0] > h[1]) std::swap(h[0], h[1]);
    int b[5];
    for (int i = 0; i < board_n; ++i) b[i] = board[i];
    std::sort(b, b + board_n);
    uint64_t key = pack_cards_key(h, b, board_n);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    if (cache_.size() >= CACHE_LIMIT) cache_.clear();
    return cache_.emplace(key, compute(h, b, board_n)).first->second;
}

}  // namespace pa
