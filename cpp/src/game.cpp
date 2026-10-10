#include "game.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cards.hpp"
#include "evaluator.hpp"

namespace pa {

namespace {

// bucketize(v, edges=(1, 3, 8)) — numpy searchsorted side="right".
inline uint8_t spr_bucket_of(double v) {
    if (v < 1.0) return 0;
    if (v < 3.0) return 1;
    if (v < 8.0) return 2;
    return 3;
}

// size_class(frac, all_in) with edges 0.5 / 0.9 / 1.25 (+1e-9).
inline Facing size_class(double frac, bool all_in) {
    if (all_in) return Facing::ALLIN;
    if (frac <= 0.5 + 1e-9) return Facing::SMALL;
    if (frac <= 0.9 + 1e-9) return Facing::MEDIUM;
    if (frac <= 1.25 + 1e-9) return Facing::LARGE;
    return Facing::OVER;
}

}  // namespace

HoldemState HoldemGame::root() const {
    HoldemState s;
    // Preflop street-start context: spr from the blinds.
    double pot = s.total[0] + s.total[1];
    double eff = cfg_.starting_stack - std::max(s.total[0], s.total[1]);
    s.spr_bucket = spr_bucket_of(pot > 0 ? eff / pot : 99.0);
    return s;
}

double HoldemGame::raise_add(int tok, double owe, double pot_now,
                             double my_street_paid) const {
    if (cfg_.is_preflop_multiple(tok)) {
        double level = my_street_paid + owe;
        return cfg_.token_value(tok) * level - my_street_paid;
    }
    return owe + cfg_.token_value(tok) * (pot_now + owe);
}

bool HoldemGame::betting_closed(const HoldemState& s) const {
    if (s.all_in) return std::fabs(s.total[0] - s.total[1]) < 1e-9;
    if (s.tokens_this_street == 0) return false;
    if (s.last_token == TOK_F) return true;
    return s.last_token == TOK_C && s.tokens_this_street >= 2;
}

bool HoldemGame::is_chance(const HoldemState& s) const {
    if (!s.dealt) return true;
    if (s.folded != -1) return false;
    return betting_closed(s) && s.board_n < 5;
}

bool HoldemGame::is_terminal(const HoldemState& s) const {
    if (!s.dealt) return false;
    if (s.folded != -1) return true;
    return s.board_n == 5 && betting_closed(s);
}

double HoldemGame::utility(const HoldemState& s) const {
    if (s.folded == 0) return -s.total[0];
    if (s.folded == 1) return s.total[1];
    int c0[7], c1[7];
    c0[0] = s.holes[0][0];
    c0[1] = s.holes[0][1];
    c1[0] = s.holes[1][0];
    c1[1] = s.holes[1][1];
    for (int i = 0; i < s.board_n; ++i) {
        c0[2 + i] = s.board[i];
        c1[2 + i] = s.board[i];
    }
    uint32_t h0 = evaluate(c0, 2 + s.board_n);
    uint32_t h1 = evaluate(c1, 2 + s.board_n);
    if (h0 > h1) return s.total[1];
    if (h1 > h0) return -s.total[0];
    return 0.0;
}

int HoldemGame::legal_actions(const HoldemState& s, int out[MAX_ACTIONS]) const {
    int me = s.to_act, opp = 1 - me;
    double owe = s.street_paid[opp] - s.street_paid[me];
    double my_stack = cfg_.starting_stack - s.total[me];
    double opp_stack = cfg_.starting_stack - s.total[opp];

    int n = 0;
    if (owe > 1e-9) out[n++] = TOK_F;
    out[n++] = TOK_C;
    if (s.n_raises < cfg_.raise_cap && my_stack > owe + 1e-9 && opp_stack > 1e-9) {
        double pot_now = s.total[0] + s.total[1];
        double min_inc = cfg_.enforce_min_raise ? s.min_inc : 0.0;
        int menu[MAX_SIZED], nm;
        cfg_.sized_menu(s.street, menu, nm);
        for (int i = 0; i < nm; ++i) {
            double add = raise_add(menu[i], owe, pot_now, s.street_paid[me]);
            if (add - owe < min_inc - 1e-9) continue;  // below NLHE minimum
            if (add < my_stack - 1e-9) out[n++] = menu[i];
        }
        out[n++] = TOK_A;
    }
    return n;
}

uint16_t HoldemGame::legal_mask(const HoldemState& s) const {
    int acts[MAX_ACTIONS];
    int n = legal_actions(s, acts);
    uint16_t m = 0;
    for (int i = 0; i < n; ++i) m |= uint16_t(1u << acts[i]);
    return m;
}

HoldemState HoldemGame::next_state(const HoldemState& s, int tok) const {
    HoldemState t = s;
    int me = t.to_act, opp = 1 - me;
    double owe = t.street_paid[opp] - t.street_paid[me];
    double my_stack = cfg_.starting_stack - t.total[me];

    double add;
    bool aggressive = false;
    Prior kind;
    if (tok == TOK_F) {
        t.folded = int8_t(me);
        add = 0.0;
        kind = Prior::FOLD;
    } else if (tok == TOK_C) {
        add = std::min(owe, my_stack);
        kind = owe > 1e-9 ? Prior::CALL : Prior::CHECK;
    } else if (tok == TOK_A) {
        add = my_stack;
        t.all_in = true;
        aggressive = true;
        kind = Prior::AGGR;
    } else {
        double pot_now = t.total[0] + t.total[1];
        add = raise_add(tok, owe, pot_now, t.street_paid[me]);
        aggressive = true;
        kind = Prior::AGGR;
    }

    if (aggressive) {
        // Betting context: size class vs pot after the call.
        double pot_after_call = t.total[0] + t.total[1] + owe;
        double frac = pot_after_call > 0 ? (add - owe) / pot_after_call : 0.0;
        t.facing[opp] = size_class(frac, tok == TOK_A);
        t.n_raises += 1;
        t.street_last_aggr = int8_t(me);
        // NLHE minimum-raise increment (HoldemGame._min_increment fold).
        if (add - owe >= t.min_inc - 1e-9) t.min_inc = add - owe;
    } else {
        t.facing[opp] = Facing::NONE;
    }
    t.own_prior[me] = kind;
    t.facing[me] = Facing::NONE;

    t.street_paid[me] += add;
    t.total[me] += add;
    t.to_act = uint8_t(opp);
    t.tokens_this_street += 1;
    t.last_token = uint8_t(tok);
    if (t.hist_n < t.hist.size()) {
        t.hist[t.hist_n++] = uint8_t(tok);
        t.street_tokens[t.street] += 1;
    } else {
        throw std::runtime_error("action history overflow");
    }
    return t;
}

void HoldemGame::start_street(HoldemState& s) const {
    if (s.street_last_aggr != -1) s.last_aggr_prev = s.street_last_aggr;
    s.street_last_aggr = -1;
    s.street += 1;
    s.street_paid = {0.0, 0.0};
    s.to_act = 1;  // big blind first postflop
    s.n_raises = 0;
    s.min_inc = BIG_BLIND;
    s.tokens_this_street = 0;
    s.last_token = 255;
    s.facing = {Facing::NONE, Facing::NONE};
    s.own_prior = {Prior::NONE, Prior::NONE};
    double pot = s.total[0] + s.total[1];
    double eff = cfg_.starting_stack - std::max(s.total[0], s.total[1]);
    s.spr_bucket = spr_bucket_of(pot > 0 ? eff / pot : 99.0);
}

HoldemState HoldemGame::with_holes(const HoldemState& s, const uint8_t holes[4]) const {
    if (s.dealt) throw std::invalid_argument("holes already dealt");
    HoldemState t = s;
    t.holes[0] = {holes[0], holes[1]};
    t.holes[1] = {holes[2], holes[3]};
    t.dealt = true;
    return t;
}

HoldemState HoldemGame::deal_street(const HoldemState& s, const uint8_t* cards,
                                    int n) const {
    HoldemState t = s;
    int need = BOARD_SIZE_AT[t.street + 1] - t.board_n;
    if (n != need) throw std::invalid_argument("wrong number of board cards");
    for (int i = 0; i < n; ++i) t.board[t.board_n++] = cards[i];
    start_street(t);
    return t;
}

HoldemState HoldemGame::sample_chance(const HoldemState& s, RandomSource& rnd) const {
    HoldemState t = s;
    if (!t.dealt) {
        // Root deal: 4 cards without replacement from 0..51 (pool order
        // ascending; tape rule floor(u * remaining) with removal).
        int pool[NUM_CARDS];
        for (int i = 0; i < NUM_CARDS; ++i) pool[i] = i;
        int picked[4];
        int size = NUM_CARDS;
        for (int k = 0; k < 4; ++k) {
            int i = rnd.below(size);
            picked[k] = pool[i];
            pool[i] = pool[size - 1];
            --size;
        }
        t.holes[0] = {uint8_t(picked[0]), uint8_t(picked[1])};
        t.holes[1] = {uint8_t(picked[2]), uint8_t(picked[3])};
        t.dealt = true;
        return t;
    }
    // Street deal: BOARD_SIZE[street+1] - board_n cards from the live deck
    // (ascending order), then enter the next street.
    bool dead[NUM_CARDS] = {};
    for (int p = 0; p < 2; ++p)
        for (int i = 0; i < 2; ++i) dead[t.holes[p][i]] = true;
    for (int i = 0; i < t.board_n; ++i) dead[t.board[i]] = true;
    int live[NUM_CARDS], nl = 0;
    for (int c = 0; c < NUM_CARDS; ++c)
        if (!dead[c]) live[nl++] = c;
    int need = BOARD_SIZE_AT[t.street + 1] - t.board_n;
    for (int k = 0; k < need; ++k) {
        int i = rnd.below(nl);
        t.board[t.board_n++] = uint8_t(live[i]);
        live[i] = live[nl - 1];
        --nl;
    }
    start_street(t);
    return t;
}

}  // namespace pa
