#include "key.hpp"

#include "cards.hpp"

namespace pa {

namespace {

inline char own_prior_char(int v) {
    switch (v) {
        case 0: return 'n';
        case 1: return 'c';
        case 2: return 'a';
        default: return 'f';
    }
}

}  // namespace

uint64_t pack_key(const HoldemGame& game, FeatureCache& features,
                  const HoldemState& s) {
    const NativeConfig& cfg = game.config();
    int player = s.to_act;
    uint64_t key = uint64_t(s.street) | (uint64_t(player) << 2);

    // Card part.
    if (s.street == 0) {
        int cid = preflop_class_id(s.holes[player][0], s.holes[player][1]);
        key |= uint64_t(cid) << 3;
    } else {
        const CardFeatures& f =
            features.get(s.holes[player].data(), s.board.data(), s.board_n);
        uint64_t sval = (cfg.river_pct_buckets > 0 && s.board_n == 5)
                            ? f.pct_bucket
                            : f.strength;
        uint64_t blocker =
            (cfg.river_blockers && s.board_n == 5) ? f.blocker : 0;
        key |= sval << 3;
        key |= uint64_t(f.draw) << 8;
        key |= uint64_t(f.nut) << 10;
        key |= blocker << 12;
        key |= uint64_t(f.texture) << 14;
    }

    // Betting context. Python's BettingContext.key() renders only
    // initiative[0], and "own"[0] == "opp"[0] == 'o': the canonical key
    // string MERGES own and opp initiative. The packed key must represent
    // the identical partition, so it stores only none=0 / some=1.
    uint64_t initiative = (s.last_aggr_prev != -1) ? 1 : 0;
    uint64_t prior;
    switch (s.own_prior[player]) {
        case Prior::NONE: prior = 0; break;
        case Prior::CHECK:
        case Prior::CALL: prior = 1; break;
        case Prior::AGGR: prior = 2; break;
        default: prior = 3; break;
    }
    key |= initiative << 18;
    key |= uint64_t(s.n_raises) << 20;
    key |= uint64_t(s.facing[player]) << 22;
    key |= prior << 25;
    key |= uint64_t(s.spr_bucket) << 27;
    key |= uint64_t(game.legal_mask(s)) << 29;
    return key;
}

std::string render_key(const NativeConfig& cfg, uint64_t key) {
    int street = int(key & 3);
    int player = int((key >> 2) & 1);
    uint64_t card = (key >> 3);

    std::string out;
    out.reserve(64);
    out += char('0' + street);
    out += '|';
    out += char('0' + player);
    out += '|';

    if (street == 0) {
        out += preflop_class_name(int(card & 0x1FF));
    } else {
        int sval = int(card & 0x1F);
        int draw = int((key >> 8) & 3);
        int nut = int((key >> 10) & 3);
        int blocker = int((key >> 12) & 3);
        int texture = int((key >> 14) & 0xF);
        bool river = street == 3;
        if (cfg.river_pct_buckets > 0 && river) {
            out += 'p';
            out += std::to_string(sval);
        } else {
            out += char('0' + sval);
        }
        out += char('0' + draw);
        out += char('0' + nut);
        if (cfg.river_blockers && river) {
            out += 'b';
            out += char('0' + blocker);
        }
        if (cfg.texture) {
            out += '|';
            out += texture_pair_char(texture);
            out += texture_suit_char(texture);
            out += texture_conn_char(texture);
        }
    }
    out += '|';

    // Betting context: i{initiative[0]}r{raises}f{facing}a{own[0]}s{spr}|legal
    int initiative = int((key >> 18) & 3);
    int raises = int((key >> 20) & 3);
    int facing = int((key >> 22) & 7);
    int prior = int((key >> 25) & 3);
    int spr = int((key >> 27) & 3);
    uint16_t mask = uint16_t((key >> 29) & 0x1FF);

    out += 'i';
    out += initiative == 0 ? 'n' : 'o';  // "none"/"own"/"opp" first letters
    out += 'r';
    out += char('0' + raises);
    out += 'f';
    out += facing_name(Facing(facing));
    out += 'a';
    out += own_prior_char(prior);
    out += 's';
    out += char('0' + spr);
    out += '|';

    // Legal tokens in Python order: f?, c, sized menu for street, a?.
    bool first = true;
    auto emit = [&](int tok) {
        if (!first) out += '.';
        first = false;
        out += cfg.token_name(tok);
    };
    if (mask & (1u << TOK_F)) emit(TOK_F);
    if (mask & (1u << TOK_C)) emit(TOK_C);
    int menu[MAX_SIZED], nm;
    cfg.sized_menu(street, menu, nm);
    for (int i = 0; i < nm; ++i)
        if (mask & (1u << menu[i])) emit(menu[i]);
    if (mask & (1u << TOK_A)) emit(TOK_A);
    return out;
}

}  // namespace pa
