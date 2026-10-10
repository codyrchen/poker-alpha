// External-sampling MCCFR — native port of poker_alpha/solvers/mccfr.py.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.hpp"
#include "features.hpp"
#include "game.hpp"
#include "key.hpp"
#include "rng.hpp"
#include "state.hpp"

namespace pa {

struct Node {
    std::array<double, MAX_ACTIONS> regret_sum{};
    std::array<double, MAX_ACTIONS> strategy_sum{};
    std::array<uint8_t, MAX_ACTIONS> actions{};  // token ids, legal order
    uint8_t num_actions = 0;
};

struct TraceEvent {
    std::string type;            // "terminal" | "chance" | "opponent" | "update"
    int player = -1;
    std::string key;
    std::vector<std::string> actions;
    std::vector<double> regrets_before;
    std::vector<double> strategy;
    int sampled = -1;            // opponent nodes
    std::vector<double> child_values;  // update nodes
    double node_value = 0.0;
    double utility = 0.0;        // terminal
};

struct SolverMetrics {
    uint64_t iterations = 0;
    uint64_t infosets = 0;
    uint64_t decision_nodes = 0;
    uint64_t chance_nodes = 0;
    uint64_t terminal_nodes = 0;
    uint64_t feature_entries = 0;
    uint64_t river_boards = 0;
    uint64_t approx_table_bytes = 0;
};

class MCCFR {
public:
    MCCFR(const NativeConfig& cfg, uint64_t seed)
        : game_(cfg), features_(cfg.river_pct_buckets), rnd_(seed) {
        index_.reserve(1 << 18);
        nodes_.reserve(1 << 18);
    }
    MCCFR(const NativeConfig& cfg, std::vector<double> tape)
        : game_(cfg), features_(cfg.river_pct_buckets), rnd_(std::move(tape)) {}

    const HoldemGame& game() const { return game_; }

    // Train `n` iterations; `should_stop` (may be empty) is polled between
    // iterations — the pybind layer uses it for Ctrl+C handling.
    void train(uint64_t n, const std::function<bool()>& should_stop = {});

    double iterate();  // one iteration (both traversals); returns p0 value

    // One traced traversal for the given update player (testing only).
    std::vector<TraceEvent> trace_traverse(int update_player);

    uint64_t iterations() const { return iterations_; }
    void set_iterations(uint64_t n) { iterations_ = n; }
    SolverMetrics metrics() const;
    size_t tape_pos() const { return rnd_.tape_pos(); }

    // State access for checkpoint/export (coarse, not hot).
    const std::unordered_map<uint64_t, uint32_t>& index() const { return index_; }
    const std::vector<Node>& nodes() const { return nodes_; }
    std::string render(uint64_t key) const { return render_key(game_.config(), key); }

    void rng_state(uint64_t out[4]) const { rnd_.get_state(out); }
    void set_rng_state(const uint64_t in[4]) { rnd_.set_state(in); }

    // Bulk-load one infoset (checkpoint restore).
    void load_infoset(uint64_t key, const uint8_t* actions, int n,
                      const double* regret, const double* strategy);

    // Force a tiny table (rehash test support).
    void shrink_table_for_tests() {
        index_.rehash(2);
        index_.max_load_factor(1.0f);
    }

private:
    uint32_t get_or_create(uint64_t key, const int* actions, int n);
    double traverse(const HoldemState& s, int update_player,
                    std::vector<TraceEvent>* trace);
    int sample_from(const double* strategy, int n);

    HoldemGame game_;
    FeatureCache features_;
    RandomSource rnd_;
    std::unordered_map<uint64_t, uint32_t> index_;
    std::vector<Node> nodes_;
    uint64_t iterations_ = 0;
    uint64_t decision_nodes_ = 0, chance_nodes_ = 0, terminal_nodes_ = 0;
};

// Regret matching (cfr.py::regret_matching semantics).
void regret_matching(const double* regrets, int n, double* out);

}  // namespace pa
