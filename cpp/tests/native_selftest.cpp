// Standalone native self-test — the sanitizer harness (Phase 51/76).
//
// macOS platform policy forbids inserting the ASan runtime into the
// hardened system/framework Python, so the Python extension cannot be
// sanitized in-process there. This executable exercises every native code
// path (rules, dealing, features, encoder, MCCFR training with forced
// rehashes, export/import, tape mode) without Python, and is built with
// ASan+UBSan by cpp/CMakeLists.txt when POKERALPHA_SANITIZE is set:
//
//   cmake -S cpp -B build-san -DPOKERALPHA_SANITIZE="address;undefined" \
//         -DPOKERALPHA_BUILD_SELFTEST=ON
//   cmake --build build-san --target native_selftest && ./build-san/native_selftest
//
// Exit code 0 = all internal invariants held and no sanitizer report.
#include <cmath>
#include <cstdio>
#include <vector>

// CHECK() vanishes under NDEBUG (RelWithDebInfo); CHECK never does.
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "CHECK failed: %s at %s:%d\n", #cond,    \
                         __FILE__, __LINE__);                              \
            return_code = 1;                                               \
            std::abort();                                                  \
        }                                                                  \
    } while (0)
static int return_code = 0;

#include "../src/cards.hpp"
#include "../src/config.hpp"
#include "../src/evaluator.hpp"
#include "../src/features.hpp"
#include "../src/game.hpp"
#include "../src/key.hpp"
#include "../src/rng.hpp"
#include "../src/solver.hpp"

using namespace pa;

static NativeConfig release_config() {
    NativeConfig cfg;
    cfg.starting_stack = 100.0;
    cfg.raise_cap = 3;
    cfg.enforce_min_raise = true;
    cfg.postflop = {{"b33", 0.33, false}, {"b75", 0.75, false}, {"b150", 1.5, false}};
    cfg.preflop = {{"x200", 2.0, true}, {"x250", 2.5, true}, {"x350", 3.5, true}};
    cfg.texture = true;
    cfg.river_blockers = true;
    cfg.river_pct_buckets = 20;
    cfg.config_signature = "selftest";
    cfg.game_signature = "selftest-game";
    cfg.encoder_signature = "selftest-encoder";
    cfg.validate();
    return cfg;
}

// Random legal playouts with internal invariants (rules fuzz).
static void fuzz_playouts(const NativeConfig& cfg, int hands, uint64_t seed) {
    HoldemGame game(cfg);
    FeatureCache features(cfg.river_pct_buckets);
    RandomSource rnd(seed);
    for (int h = 0; h < hands; ++h) {
        HoldemState s = game.root();
        int guard = 0;
        while (!game.is_terminal(s)) {
            CHECK(++guard < 200);
            if (game.is_chance(s)) {
                s = game.sample_chance(s, rnd);
                // No duplicate cards.
                bool seen[NUM_CARDS] = {};
                for (int p = 0; p < 2; ++p)
                    for (int i = 0; i < 2; ++i) {
                        CHECK(!seen[s.holes[p][i]]);
                        seen[s.holes[p][i]] = true;
                    }
                for (int i = 0; i < s.board_n; ++i) {
                    CHECK(!seen[s.board[i]]);
                    seen[s.board[i]] = true;
                }
                continue;
            }
            int actions[MAX_ACTIONS];
            int n = game.legal_actions(s, actions);
            CHECK(n >= 1 && n <= MAX_ACTIONS);
            uint64_t key = pack_key(game, features, s);
            std::string rendered = render_key(cfg, key);
            CHECK(!rendered.empty());
            s = game.next_state(s, actions[int(rnd.below(n))]);
            CHECK(s.total[0] <= cfg.starting_stack + 1e-9);
            CHECK(s.total[1] <= cfg.starting_stack + 1e-9);
        }
        double u = game.utility(s);
        CHECK(std::isfinite(u));
        CHECK(std::fabs(u) <= cfg.starting_stack + 1e-9);
    }
    std::printf("fuzz_playouts: %d hands ok\n", hands);
}

static void train_with_rehash(const NativeConfig& cfg) {
    MCCFR solver(cfg, 7);
    solver.shrink_table_for_tests();   // force many rehashes while training
    solver.train(1500);
    SolverMetrics m = solver.metrics();
    CHECK(m.iterations == 1500);
    CHECK(m.infosets > 10'000);
    // Finiteness audit of every accumulator (Phase 77).
    for (const Node& nd : solver.nodes()) {
        for (int i = 0; i < nd.num_actions; ++i) {
            CHECK(std::isfinite(nd.regret_sum[size_t(i)]));
            CHECK(std::isfinite(nd.strategy_sum[size_t(i)]));
            CHECK(nd.strategy_sum[size_t(i)] >= 0.0);
        }
    }
    std::printf("train_with_rehash: %llu infosets ok\n",
                (unsigned long long)m.infosets);

    // Export/import roundtrip through load_infoset.
    MCCFR copy(cfg, 7);
    for (const auto& [key, idx] : solver.index()) {
        const Node& nd = solver.nodes()[idx];
        copy.load_infoset(key, nd.actions.data(), nd.num_actions,
                          nd.regret_sum.data(), nd.strategy_sum.data());
    }
    CHECK(copy.nodes().size() == solver.nodes().size());
    std::printf("export/import roundtrip ok\n");
}

static void tape_mode(const NativeConfig& cfg) {
    Xoshiro256 gen(42);
    std::vector<double> tape(200 * 4000);
    for (auto& x : tape) x = gen.next_double();
    MCCFR a(cfg, tape);
    MCCFR b(cfg, tape);
    for (int i = 0; i < 200; ++i) {
        double va = a.iterate();
        double vb = b.iterate();
        CHECK(va == vb);
    }
    CHECK(a.tape_pos() == b.tape_pos());
    std::printf("tape determinism: 200 iterations ok (pos %zu)\n", a.tape_pos());
}

static void evaluator_spot_checks() {
    // Royal flush > quads; wheel == wheel; ties on the board.
    int royal[5] = {8 + 39, 9 + 39, 10 + 39, 11 + 39, 12 + 39};
    int quads[5] = {12, 25, 38, 51, 0};
    CHECK(evaluate(royal, 5) > evaluate(quads, 5));
    int wheel1[5] = {12, 0, 1, 2, 3};
    int wheel2[5] = {25, 13, 14, 15, 16};
    CHECK(evaluate(wheel1, 5) == evaluate(wheel2, 5));
    std::printf("evaluator spot checks ok\n");
}

int main() {
    NativeConfig cfg = release_config();
    evaluator_spot_checks();
    fuzz_playouts(cfg, 20'000, 1);
    train_with_rehash(cfg);
    tape_mode(cfg);
    // v1-style config (no preflop multiples, no min raise) shares paths.
    NativeConfig v1 = cfg;
    v1.preflop.clear();
    v1.enforce_min_raise = false;
    v1.river_pct_buckets = 0;
    fuzz_playouts(v1, 8'000, 2);
    {
        MCCFR s(v1, 3);
        s.train(400);
        CHECK(s.metrics().infosets > 5'000);
    }
    std::printf("NATIVE SELFTEST PASS\n");
    return 0;
}
