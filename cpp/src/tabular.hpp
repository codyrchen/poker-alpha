// Tabular-tree MCCFR — the minimal native exact-validation adapter
// (final-trust project, Phase 3).
//
// The Python side flattens any root-chance two-player zero-sum Game
// (reduced preflop, fixed-river subgames) into: a betting tree shared by
// all deals, a weighted deal list over (hand0, hand1), and a dense
// per-terminal utility matrix u[terminal][h0][h1] (player-0 utility). The
// native solver then runs the exact same external-sampling MCCFR update
// rules as the Hold'em backend (regret matching, cdf sampling,
// exactly-rounded strategy dot, uniform average-strategy tallies) on that
// tree, so exact reduced-game convergence is measured DIRECTLY on native
// code rather than transitively through tape parity.
//
// This is deliberately NOT a generic C++ game framework: one chance node
// (the root deal), perfect-recall keys (betting node x own hand).
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "rng.hpp"
#include "solver.hpp"  // Node, MAX_ACTIONS, regret_matching

namespace pa {

struct TabularTree {
    // Nodes: player[i] in {0,1} for decision nodes, -1 for terminals.
    std::vector<int8_t> player;
    std::vector<uint32_t> child_off;   // per node, into children (decision only)
    std::vector<uint32_t> children;
    std::vector<int32_t> term_idx;     // per node: utility-matrix row, -1 if decision
    // Deals.
    std::vector<double> deal_cdf;      // normalized cumulative weights
    std::vector<uint16_t> deal_h0, deal_h1;
    int n_h0 = 0, n_h1 = 0;
    // util[term_idx * n_h0 * n_h1 + h0 * n_h1 + h1] = player-0 utility.
    std::vector<double> util;

    int num_actions(uint32_t node) const {
        return int(child_off[node + 1] - child_off[node]);
    }
};

class TabularMCCFR {
public:
    TabularMCCFR(TabularTree tree, uint64_t seed)
        : tree_(std::move(tree)), rnd_(seed) {}

    void train(uint64_t n) {
        for (uint64_t t = 0; t < n; ++t) iterate();
    }

    double iterate();
    uint64_t iterations() const { return iterations_; }

    // (node, hand, average strategy) for every infoset.
    struct Row {
        uint32_t node;
        uint32_t hand;
        std::vector<double> probs;
        double visits;   // strategy_sum total (non-updating-player visits)
    };
    std::vector<Row> average_strategy() const;

private:
    double traverse(uint32_t node, int h0, int h1, int update_player);
    int sample_from(const double* strategy, int n);
    uint32_t get_or_create(uint64_t key, int n);

    TabularTree tree_;
    RandomSource rnd_;
    std::unordered_map<uint64_t, uint32_t> index_;
    std::vector<Node> nodes_;
    uint64_t iterations_ = 0;
};

}  // namespace pa
