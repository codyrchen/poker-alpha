#include "tabular.hpp"

#include <algorithm>

#include "fsum.hpp"

namespace pa {

int TabularMCCFR::sample_from(const double* strategy, int n) {
    // Identical semantics to MCCFR::sample_from (MCCFRSolver._sample).
    double cdf[MAX_ACTIONS];
    double cum = 0.0;
    for (int i = 0; i < n; ++i) {
        cum += strategy[i];
        cdf[i] = cum;
    }
    double last = cdf[n - 1];
    for (int i = 0; i < n; ++i) cdf[i] /= last;
    double u = rnd_.uniform();
    int idx = int(std::upper_bound(cdf, cdf + n, u) - cdf);
    return idx >= n ? n - 1 : idx;
}

uint32_t TabularMCCFR::get_or_create(uint64_t key, int n) {
    auto it = index_.find(key);
    if (it != index_.end()) return it->second;
    Node node;
    node.num_actions = uint8_t(n);
    for (int i = 0; i < n; ++i) node.actions[size_t(i)] = uint8_t(i);
    uint32_t idx = uint32_t(nodes_.size());
    nodes_.push_back(node);
    index_.emplace(key, idx);
    return idx;
}

double TabularMCCFR::traverse(uint32_t node, int h0, int h1, int update_player) {
    int player = tree_.player[node];
    if (player < 0) {  // terminal
        double u0 = tree_.util[size_t(tree_.term_idx[node]) *
                                   size_t(tree_.n_h0) * size_t(tree_.n_h1) +
                               size_t(h0) * size_t(tree_.n_h1) + size_t(h1)];
        return update_player == 0 ? u0 : -u0;
    }
    int n = tree_.num_actions(node);
    int own = player == 0 ? h0 : h1;
    uint64_t key = (uint64_t(node) << 16) | uint64_t(own);
    uint32_t idx = get_or_create(key, n);

    double strategy[MAX_ACTIONS];
    double regrets[MAX_ACTIONS];
    for (int i = 0; i < n; ++i) regrets[i] = nodes_[idx].regret_sum[size_t(i)];
    regret_matching(regrets, n, strategy);

    uint32_t base = tree_.child_off[node];
    if (player != update_player) {
        Node& nd = nodes_[idx];
        for (int i = 0; i < n; ++i) nd.strategy_sum[size_t(i)] += strategy[i];
        int pick = sample_from(strategy, n);
        return traverse(tree_.children[base + uint32_t(pick)], h0, h1,
                        update_player);
    }

    double child_values[MAX_ACTIONS];
    for (int i = 0; i < n; ++i)
        child_values[i] = traverse(tree_.children[base + uint32_t(i)], h0, h1,
                                   update_player);
    double node_value = strategy_dot(strategy, child_values, size_t(n));
    Node& nd = nodes_[idx];  // re-index: children may have inserted
    for (int i = 0; i < n; ++i)
        nd.regret_sum[size_t(i)] += child_values[i] - node_value;
    return node_value;
}

double TabularMCCFR::iterate() {
    double value = 0.0;
    for (int up = 0; up < 2; ++up) {
        // Root chance: sample one deal from the normalized cdf.
        double u = rnd_.uniform();
        size_t i = size_t(std::upper_bound(tree_.deal_cdf.begin(),
                                           tree_.deal_cdf.end(), u) -
                          tree_.deal_cdf.begin());
        if (i >= tree_.deal_cdf.size()) i = tree_.deal_cdf.size() - 1;
        double v = traverse(0, tree_.deal_h0[i], tree_.deal_h1[i], up);
        if (up == 0) value = v;
    }
    iterations_ += 1;
    return value;
}

std::vector<TabularMCCFR::Row> TabularMCCFR::average_strategy() const {
    std::vector<Row> out;
    out.reserve(index_.size());
    for (const auto& [key, idx] : index_) {
        const Node& nd = nodes_[idx];
        Row row;
        row.node = uint32_t(key >> 16);
        row.hand = uint32_t(key & 0xFFFF);
        double total = 0.0;
        for (int i = 0; i < nd.num_actions; ++i)
            total += nd.strategy_sum[size_t(i)];
        row.probs.resize(size_t(nd.num_actions));
        for (int i = 0; i < nd.num_actions; ++i)
            row.probs[size_t(i)] = total > 0.0
                                       ? nd.strategy_sum[size_t(i)] / total
                                       : 1.0 / nd.num_actions;
        row.visits = total;
        out.push_back(std::move(row));
    }
    return out;
}

}  // namespace pa
