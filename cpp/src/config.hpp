// Native solver configuration — transported from HoldemSolverConfig
// (poker_alpha/solver_config.py). Python remains the source of truth; the
// native side validates it supports the combination and stores the
// signature strings verbatim for checkpoint stamping.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pa {

constexpr double SMALL_BLIND = 0.5;
constexpr double BIG_BLIND = 1.0;
constexpr int MAX_ACTIONS = 6;      // f, c, up to 3 sized raises, a
constexpr int MAX_SIZED = 3;        // sized tokens per street menu
constexpr int MAX_TOKENS = 3 + 2 * MAX_SIZED;  // f,c,a + postflop + preflop

// Global action-token ids. 0..2 fixed; sized tokens follow in config order.
enum : uint8_t { TOK_F = 0, TOK_C = 1, TOK_A = 2, TOK_SIZED0 = 3 };

struct SizedToken {
    std::string name;   // e.g. "b33" or "x200"
    double value;       // pot fraction (b) or raise-to multiple (x)
    bool preflop_multiple;  // true for x tokens
};

struct NativeConfig {
    double starting_stack = 100.0;
    int raise_cap = 3;
    bool enforce_min_raise = false;
    std::vector<SizedToken> postflop;   // bet fractions, config order
    std::vector<SizedToken> preflop;    // raise-to multiples (may be empty = v1)
    // Encoder (CompactHoldemEncoder, history="abstract" only):
    bool texture = true;
    bool river_blockers = true;
    int river_pct_buckets = 0;          // 0 = strength ladder on the river
    // Signatures, stored verbatim (checkpoint stamping):
    std::string config_signature;
    std::string game_signature;
    std::string encoder_signature;

    void validate() const {
        if (postflop.empty() || postflop.size() > MAX_SIZED)
            throw std::invalid_argument("postflop bet menu must have 1..3 entries");
        if (preflop.size() > MAX_SIZED)
            throw std::invalid_argument("preflop raise menu must have 0..3 entries");
        if (raise_cap < 1 || starting_stack <= 0)
            throw std::invalid_argument("invalid tree parameters");
        if (river_pct_buckets < 0 || river_pct_buckets > 20)
            throw std::invalid_argument("river_pct_buckets must be 0..20");
    }

    // Global token table: f, c, a, postflop..., preflop...
    int num_tokens() const { return 3 + int(postflop.size()) + int(preflop.size()); }
    int preflop_base() const { return 3 + int(postflop.size()); }

    const std::string& token_name(int tok) const {
        static const std::string F = "f", C = "c", A = "a";
        if (tok == TOK_F) return F;
        if (tok == TOK_C) return C;
        if (tok == TOK_A) return A;
        int i = tok - TOK_SIZED0;
        if (i < int(postflop.size())) return postflop[size_t(i)].name;
        return preflop[size_t(i - int(postflop.size()))].name;
    }

    double token_value(int tok) const {
        int i = tok - TOK_SIZED0;
        if (i < int(postflop.size())) return postflop[size_t(i)].value;
        return preflop[size_t(i - int(postflop.size()))].value;
    }

    bool is_preflop_multiple(int tok) const {
        return tok >= preflop_base();
    }

    // The sized menu used on `street` (token ids, config order). Mirrors
    // HoldemGame.raise_tokens: preflop multiples on street 0 when present.
    void sized_menu(int street, int out[MAX_SIZED], int& n) const {
        n = 0;
        if (street == 0 && !preflop.empty()) {
            for (size_t i = 0; i < preflop.size(); ++i)
                out[n++] = preflop_base() + int(i);
        } else {
            for (size_t i = 0; i < postflop.size(); ++i)
                out[n++] = TOK_SIZED0 + int(i);
        }
    }

    int token_from_name(const std::string& name) const {
        for (int t = 0; t < num_tokens(); ++t)
            if (token_name(t) == name) return t;
        throw std::invalid_argument("unknown action token: " + name);
    }
};

}  // namespace pa
