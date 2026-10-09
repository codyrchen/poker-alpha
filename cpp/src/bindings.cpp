// pybind11 bindings for the PokerAlpha native MCCFR backend.
//
// The production surface is NativeSolverCore: construct once, train() in
// coarse chunks with the GIL released, and move state in/out as bulk arrays
// for Python-side checkpoint/artifact writing. The debug_* functions exist
// for the cross-language parity test suite and are not a supported
// execution path for training.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstring>

#include "cards.hpp"
#include "config.hpp"
#include "evaluator.hpp"
#include "features.hpp"
#include "fsum.hpp"
#include "game.hpp"
#include "key.hpp"
#include "rng.hpp"
#include "solver.hpp"
#include "state.hpp"

namespace py = pybind11;
using namespace pa;

namespace {

NativeConfig config_from_kwargs(double starting_stack, int raise_cap,
                                bool enforce_min_raise,
                                const std::vector<std::pair<std::string, double>>& postflop,
                                const std::vector<std::pair<std::string, double>>& preflop,
                                bool texture, bool river_blockers,
                                int river_pct_buckets,
                                const std::string& config_signature,
                                const std::string& game_signature,
                                const std::string& encoder_signature) {
    NativeConfig cfg;
    cfg.starting_stack = starting_stack;
    cfg.raise_cap = raise_cap;
    cfg.enforce_min_raise = enforce_min_raise;
    for (const auto& [name, value] : postflop)
        cfg.postflop.push_back({name, value, false});
    for (const auto& [name, value] : preflop)
        cfg.preflop.push_back({name, value, true});
    cfg.texture = texture;
    cfg.river_blockers = river_blockers;
    cfg.river_pct_buckets = river_pct_buckets;
    cfg.config_signature = config_signature;
    cfg.game_signature = game_signature;
    cfg.encoder_signature = encoder_signature;
    cfg.validate();
    return cfg;
}

// Replay a Python trajectory into a native state (parity tests).
HoldemState make_state(const HoldemGame& game,
                       const std::vector<int>& holes,
                       const std::vector<int>& board,
                       const std::vector<std::vector<std::string>>& streets) {
    HoldemState s = game.root();
    if (!holes.empty()) {
        if (holes.size() != 4) throw std::invalid_argument("need 4 hole cards");
        uint8_t h[4];
        for (int i = 0; i < 4; ++i) h[i] = uint8_t(holes[size_t(i)]);
        s = game.with_holes(s, h);
    }
    size_t bpos = 0;
    for (size_t si = 0; si < streets.size(); ++si) {
        if (si > 0) {
            int need = BOARD_SIZE_AT[si] - s.board_n;
            if (bpos + size_t(need) > board.size())
                throw std::invalid_argument("not enough board cards");
            uint8_t cards[3];
            for (int i = 0; i < need; ++i)
                cards[i] = uint8_t(board[bpos++]);
            s = game.deal_street(s, cards, need);
        }
        for (const auto& tok : streets[si])
            s = game.next_state(s, game.config().token_from_name(tok));
    }
    return s;
}

py::dict state_info(const HoldemGame& game, FeatureCache& features,
                    const HoldemState& s) {
    py::dict d;
    d["is_chance"] = game.is_chance(s);
    d["is_terminal"] = game.is_terminal(s);
    d["current_player"] = int(s.to_act);
    d["street"] = int(s.street);
    d["pot"] = s.pot();
    d["contrib"] = py::make_tuple(s.total[0], s.total[1]);
    d["folded"] = int(s.folded);
    d["all_in"] = s.all_in;
    if (game.is_terminal(s)) d["utility"] = game.utility(s);
    if (!game.is_terminal(s) && !game.is_chance(s)) {
        int acts[MAX_ACTIONS];
        int n = game.legal_actions(s, acts);
        py::list legal;
        for (int i = 0; i < n; ++i)
            legal.append(game.config().token_name(acts[i]));
        d["legal_actions"] = legal;
        uint64_t key = pack_key(game, features, s);
        d["key_u64"] = key;
        d["infoset_key"] = render_key(game.config(), key);
    }
    return d;
}

py::dict trace_event_dict(const TraceEvent& e) {
    py::dict d;
    d["type"] = e.type;
    if (e.type == "terminal") {
        d["utility"] = e.utility;
        return d;
    }
    if (e.type == "chance") return d;
    d["player"] = e.player;
    d["key"] = e.key;
    d["actions"] = e.actions;
    d["regrets_before"] = e.regrets_before;
    d["strategy"] = e.strategy;
    if (e.type == "opponent") d["sampled"] = e.sampled;
    if (e.type == "update") {
        d["child_values"] = e.child_values;
        d["node_value"] = e.node_value;
    }
    return d;
}

class NativeSolverCore {
public:
    NativeSolverCore(const NativeConfig& cfg, uint64_t seed)
        : solver_(cfg, seed) {}
    NativeSolverCore(const NativeConfig& cfg, std::vector<double> tape)
        : solver_(cfg, std::move(tape)) {}

    void train(uint64_t n, uint64_t chunk) {
        if (chunk == 0) chunk = 2000;
        uint64_t done = 0;
        while (done < n) {
            uint64_t take = std::min(chunk, n - done);
            {
                py::gil_scoped_release release;
                solver_.train(take);
            }
            done += take;
            if (PyErr_CheckSignals() != 0) throw py::error_already_set();
        }
    }

    double iterate() { return solver_.iterate(); }
    uint64_t iterations() const { return solver_.iterations(); }
    void set_iterations(uint64_t n) { solver_.set_iterations(n); }
    size_t tape_pos() const { return solver_.tape_pos(); }

    py::dict metrics() const {
        SolverMetrics m = solver_.metrics();
        py::dict d;
        d["iterations"] = m.iterations;
        d["infosets"] = m.infosets;
        d["decision_nodes"] = m.decision_nodes;
        d["chance_nodes"] = m.chance_nodes;
        d["terminal_nodes"] = m.terminal_nodes;
        d["feature_cache_entries"] = m.feature_entries;
        d["river_board_cache_entries"] = m.river_boards;
        d["approx_table_bytes"] = m.approx_table_bytes;
        return d;
    }

    py::list trace_traverse(int update_player) {
        py::list out;
        for (const auto& e : solver_.trace_traverse(update_player))
            out.append(trace_event_dict(e));
        return out;
    }

    // Bulk state export: (keys_u64, keys_str, offsets, action_tokens,
    // regret, strategy, rng_state). Keys are sorted by rendered UTF-8 string
    // (the checkpoint / artifact canonical order).
    py::tuple export_state() const {
        const auto& index = solver_.index();
        const auto& nodes = solver_.nodes();
        std::vector<std::pair<std::string, uint64_t>> order;
        order.reserve(index.size());
        for (const auto& [key, idx] : index) {
            (void)idx;
            order.emplace_back(solver_.render(key), key);
        }
        std::sort(order.begin(), order.end());

        size_t n = order.size();
        py::array_t<uint64_t> keys_u64(static_cast<py::ssize_t>(n));
        py::list keys_str;
        std::vector<int64_t> offsets;
        offsets.reserve(n + 1);
        offsets.push_back(0);
        std::vector<uint8_t> tokens;
        std::vector<double> regret, strategy;
        auto* ku = keys_u64.mutable_data();
        for (size_t i = 0; i < n; ++i) {
            const auto& [str, key] = order[i];
            ku[i] = key;
            keys_str.append(str);
            const Node& nd = nodes[index.at(key)];
            for (int a = 0; a < nd.num_actions; ++a) {
                tokens.push_back(nd.actions[size_t(a)]);
                regret.push_back(nd.regret_sum[size_t(a)]);
                strategy.push_back(nd.strategy_sum[size_t(a)]);
            }
            offsets.push_back(int64_t(tokens.size()));
        }
        uint64_t rng[4];
        solver_.rng_state(rng);
        py::array_t<uint64_t> rng_arr(4);
        std::memcpy(rng_arr.mutable_data(), rng, sizeof(rng));
        return py::make_tuple(
            keys_u64, keys_str,
            py::array_t<int64_t>(py::ssize_t(offsets.size()), offsets.data()),
            py::array_t<uint8_t>(py::ssize_t(tokens.size()), tokens.data()),
            py::array_t<double>(py::ssize_t(regret.size()), regret.data()),
            py::array_t<double>(py::ssize_t(strategy.size()), strategy.data()),
            rng_arr);
    }

    void import_state(py::array_t<uint64_t> keys_u64,
                      py::array_t<int64_t> offsets,
                      py::array_t<uint8_t> tokens,
                      py::array_t<double> regret,
                      py::array_t<double> strategy,
                      py::array_t<uint64_t> rng_state, uint64_t iterations) {
        auto k = keys_u64.unchecked<1>();
        auto off = offsets.unchecked<1>();
        auto tok = tokens.unchecked<1>();
        auto rg = regret.unchecked<1>();
        auto st = strategy.unchecked<1>();
        if (off.shape(0) != k.shape(0) + 1)
            throw std::invalid_argument("inconsistent offsets");
        if (tok.shape(0) != rg.shape(0) || tok.shape(0) != st.shape(0))
            throw std::invalid_argument("inconsistent array lengths");
        if (off.shape(0) > 0 && (off(0) != 0 || off(off.shape(0) - 1) != tok.shape(0)))
            throw std::invalid_argument("inconsistent offsets");
        for (py::ssize_t i = 0; i < k.shape(0); ++i) {
            int64_t lo = off(i), hi = off(i + 1);
            if (hi <= lo || hi - lo > MAX_ACTIONS)
                throw std::invalid_argument("inconsistent action layout");
            solver_.load_infoset(k(i), tok.data(size_t(lo)), int(hi - lo),
                                 rg.data(size_t(lo)), st.data(size_t(lo)));
        }
        if (rng_state.size() != 4)
            throw std::invalid_argument("rng_state must have 4 words");
        uint64_t rng[4];
        std::memcpy(rng, rng_state.data(), sizeof(rng));
        solver_.set_rng_state(rng);
        solver_.set_iterations(iterations);
    }

    std::string render_key_u64(uint64_t key) const { return solver_.render(key); }
    void shrink_table_for_tests() { solver_.shrink_table_for_tests(); }

    const HoldemGame& game() const { return solver_.game(); }

private:
    MCCFR solver_;
};

}  // namespace

PYBIND11_MODULE(poker_alpha_native, m) {
    m.doc() = "PokerAlpha native MCCFR backend (C++17, external sampling)";
    m.attr("__version__") = "0.1.0";
    m.attr("BACKEND_SCHEMA") = 1;
    m.attr("RNG_NAME") = "xoshiro256**/v1";

    py::class_<NativeConfig>(m, "NativeConfig")
        .def(py::init(&config_from_kwargs), py::arg("starting_stack"),
             py::arg("raise_cap"), py::arg("enforce_min_raise"),
             py::arg("postflop"), py::arg("preflop"), py::arg("texture"),
             py::arg("river_blockers"), py::arg("river_pct_buckets"),
             py::arg("config_signature"), py::arg("game_signature"),
             py::arg("encoder_signature"))
        .def_readonly("config_signature", &NativeConfig::config_signature)
        .def_readonly("game_signature", &NativeConfig::game_signature)
        .def_readonly("encoder_signature", &NativeConfig::encoder_signature);

    py::class_<NativeSolverCore>(m, "NativeSolverCore")
        .def(py::init<const NativeConfig&, uint64_t>(), py::arg("config"),
             py::arg("seed"))
        .def(py::init<const NativeConfig&, std::vector<double>>(),
             py::arg("config"), py::arg("tape"))
        .def("train", &NativeSolverCore::train, py::arg("iterations"),
             py::arg("chunk") = 2000)
        .def("iterate", &NativeSolverCore::iterate)
        .def_property_readonly("iterations", &NativeSolverCore::iterations)
        .def("metrics", &NativeSolverCore::metrics)
        .def("trace_traverse", &NativeSolverCore::trace_traverse,
             py::arg("update_player"))
        .def("export_state", &NativeSolverCore::export_state)
        .def("import_state", &NativeSolverCore::import_state,
             py::arg("keys_u64"), py::arg("offsets"), py::arg("tokens"),
             py::arg("regret"), py::arg("strategy"), py::arg("rng_state"),
             py::arg("iterations"))
        .def("render_key_u64", &NativeSolverCore::render_key_u64)
        .def("tape_pos", &NativeSolverCore::tape_pos)
        .def("shrink_table_for_tests", &NativeSolverCore::shrink_table_for_tests);

    // ---- debug / parity API (tests only; not a training path) ----------
    m.def("debug_evaluate",
          [](const std::vector<int>& cards) {
              if (cards.size() < 5 || cards.size() > 7)
                  throw std::invalid_argument("need 5-7 cards");
              for (int c : cards)
                  if (c < 0 || c >= NUM_CARDS)
                      throw std::invalid_argument("card code out of range");
              return evaluate(cards.data(), int(cards.size()));
          },
          "Packed uint32 hand value of 5-7 card codes");

    m.def("debug_state_info",
          [](const NativeConfig& cfg, const std::vector<int>& holes,
             const std::vector<int>& board,
             const std::vector<std::vector<std::string>>& streets) {
              HoldemGame game(cfg);
              FeatureCache features(cfg.river_pct_buckets);
              HoldemState s = make_state(game, holes, board, streets);
              return state_info(game, features, s);
          },
          py::arg("config"), py::arg("holes"), py::arg("board"),
          py::arg("streets"),
          "Replay a Python trajectory and report the native state view");

    m.def("debug_features",
          [](const std::vector<int>& hole, const std::vector<int>& board,
             int river_pct_buckets) {
              if (hole.size() != 2 || board.size() < 3 || board.size() > 5)
                  throw std::invalid_argument("need 2 hole + 3-5 board cards");
              FeatureCache fc(river_pct_buckets);
              uint8_t h[2] = {uint8_t(hole[0]), uint8_t(hole[1])};
              uint8_t b[5];
              for (size_t i = 0; i < board.size(); ++i) b[i] = uint8_t(board[i]);
              const CardFeatures& f = fc.get(h, b, int(board.size()));
              py::dict d;
              d["strength"] = int(f.strength);
              d["draw"] = int(f.draw);
              d["nut"] = int(f.nut);
              d["blocker"] = int(f.blocker);
              std::string tex;
              tex += texture_pair_char(f.texture);
              tex += texture_suit_char(f.texture);
              tex += texture_conn_char(f.texture);
              d["texture"] = tex;
              d["pct_bucket"] = int(f.pct_bucket);
              return d;
          },
          py::arg("hole"), py::arg("board"), py::arg("river_pct_buckets") = 0);

    m.def("debug_preflop_class",
          [](int a, int b) { return preflop_class_name(preflop_class_id(a, b)); });

    m.def("debug_deal_counts",
          [](const NativeConfig& cfg, int hands, uint64_t seed) {
              // Statistical sanity of production-RNG dealing (Phase 9):
              // counts of every card over `hands` root deals, per seat slot.
              HoldemGame game(cfg);
              RandomSource rnd(seed);
              py::array_t<int64_t> counts({4, NUM_CARDS});
              auto c = counts.mutable_unchecked<2>();
              for (py::ssize_t i = 0; i < 4; ++i)
                  for (py::ssize_t j = 0; j < NUM_CARDS; ++j) c(i, j) = 0;
              for (int h = 0; h < hands; ++h) {
                  HoldemState s = game.sample_chance(game.root(), rnd);
                  c(0, s.holes[0][0]) += 1;
                  c(1, s.holes[0][1]) += 1;
                  c(2, s.holes[1][0]) += 1;
                  c(3, s.holes[1][1]) += 1;
              }
              return counts;
          },
          py::arg("config"), py::arg("hands"), py::arg("seed"));

    m.def("debug_fsum", [](const std::vector<double>& xs) {
        if (xs.size() > 8) throw std::invalid_argument("max 8 terms");
        return fsum<8>(xs.data(), xs.size());
    });
}
