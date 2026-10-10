#include "solver.hpp"

#include <algorithm>
#include <stdexcept>

#include "fsum.hpp"

namespace pa {

void regret_matching(const double* regrets, int n, double* out) {
    double total = 0.0;
    double positive[MAX_ACTIONS];
    for (int i = 0; i < n; ++i) {
        positive[i] = regrets[i] > 0.0 ? regrets[i] : 0.0;
        total += positive[i];  // sequential, like numpy's small-array sum
    }
    if (total > 0.0) {
        for (int i = 0; i < n; ++i) out[i] = positive[i] / total;
    } else {
        double u = 1.0 / n;
        for (int i = 0; i < n; ++i) out[i] = u;
    }
}

int MCCFR::sample_from(const double* strategy, int n) {
    // MCCFRSolver._sample: cumsum, divide by last, searchsorted right.
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

uint32_t MCCFR::get_or_create(uint64_t key, const int* actions, int n) {
    auto it = index_.find(key);
    if (it != index_.end()) return it->second;
    Node node;
    node.num_actions = uint8_t(n);
    for (int i = 0; i < n; ++i) node.actions[size_t(i)] = uint8_t(actions[i]);
    uint32_t idx = uint32_t(nodes_.size());
    nodes_.push_back(node);
    index_.emplace(key, idx);
    return idx;
}

double MCCFR::traverse(const HoldemState& s, int update_player,
                       std::vector<TraceEvent>* trace) {
    if (game_.is_terminal(s)) {
        ++terminal_nodes_;
        double u0 = game_.utility(s);
        double u = update_player == 0 ? u0 : -u0;
        if (trace) {
            TraceEvent e;
            e.type = "terminal";
            e.utility = u;
            trace->push_back(e);
        }
        return u;
    }
    if (game_.is_chance(s)) {
        ++chance_nodes_;
        if (trace) {
            TraceEvent e;
            e.type = "chance";
            trace->push_back(e);
        }
        return traverse(game_.sample_chance(s, rnd_), update_player, trace);
    }

    ++decision_nodes_;
    int player = s.to_act;
    uint64_t key = pack_key(game_, features_, s);
    int actions[MAX_ACTIONS];
    int n = game_.legal_actions(s, actions);
    uint32_t idx = get_or_create(key, actions, n);

    double strategy[MAX_ACTIONS];
    double regrets_before[MAX_ACTIONS];
    for (int i = 0; i < n; ++i) regrets_before[i] = nodes_[idx].regret_sum[size_t(i)];
    regret_matching(regrets_before, n, strategy);

    auto fill_trace = [&](TraceEvent& e) {
        e.player = player;
        e.key = render_key(game_.config(), key);
        for (int i = 0; i < n; ++i) {
            e.actions.push_back(game_.config().token_name(actions[i]));
            e.regrets_before.push_back(regrets_before[i]);
            e.strategy.push_back(strategy[i]);
        }
    };

    if (player != update_player) {
        // Opponent node: tally average strategy, sample one action.
        Node& nd = nodes_[idx];
        for (int i = 0; i < n; ++i) nd.strategy_sum[size_t(i)] += strategy[i];
        int pick = sample_from(strategy, n);
        if (trace) {
            TraceEvent e;
            e.type = "opponent";
            fill_trace(e);
            e.sampled = pick;
            trace->push_back(e);
        }
        return traverse(game_.next_state(s, actions[pick]), update_player, trace);
    }

    // Updating player's node: explore every action. Child recursion can
    // insert infosets (growing nodes_ / rehashing index_), so no Node
    // reference is held across it — the node is re-indexed afterwards.
    double child_values[MAX_ACTIONS];
    for (int i = 0; i < n; ++i)
        child_values[i] = traverse(game_.next_state(s, actions[i]),
                                   update_player, trace);
    double node_value = strategy_dot(strategy, child_values, size_t(n));
    Node& nd = nodes_[idx];
    for (int i = 0; i < n; ++i)
        nd.regret_sum[size_t(i)] += child_values[i] - node_value;
    if (trace) {
        TraceEvent e;
        e.type = "update";
        fill_trace(e);
        for (int i = 0; i < n; ++i) e.child_values.push_back(child_values[i]);
        e.node_value = node_value;
        trace->push_back(e);
    }
    return node_value;
}

double MCCFR::iterate() {
    double value = traverse(game_.root(), 0, nullptr);
    traverse(game_.root(), 1, nullptr);
    iterations_ += 1;
    return value;
}

void MCCFR::train(uint64_t n, const std::function<bool()>& should_stop) {
    for (uint64_t t = 0; t < n; ++t) {
        if (should_stop && should_stop()) return;
        iterate();
    }
}

std::vector<TraceEvent> MCCFR::trace_traverse(int update_player) {
    std::vector<TraceEvent> trace;
    traverse(game_.root(), update_player, &trace);
    return trace;
}

SolverMetrics MCCFR::metrics() const {
    SolverMetrics m;
    m.iterations = iterations_;
    m.infosets = nodes_.size();
    m.decision_nodes = decision_nodes_;
    m.chance_nodes = chance_nodes_;
    m.terminal_nodes = terminal_nodes_;
    m.feature_entries = features_.feature_entries();
    m.river_boards = features_.river_board_entries();
    m.approx_table_bytes =
        nodes_.size() * sizeof(Node) +
        index_.bucket_count() * (sizeof(uint64_t) + sizeof(uint32_t) + 8);
    return m;
}

void MCCFR::load_infoset(uint64_t key, const uint8_t* actions, int n,
                         const double* regret, const double* strategy) {
    if (n <= 0 || n > MAX_ACTIONS)
        throw std::invalid_argument("bad infoset action count");
    if (index_.count(key))
        throw std::invalid_argument("duplicate infoset key in checkpoint");
    Node node;
    node.num_actions = uint8_t(n);
    for (int i = 0; i < n; ++i) {
        node.actions[size_t(i)] = actions[i];
        node.regret_sum[size_t(i)] = regret[i];
        node.strategy_sum[size_t(i)] = strategy[i];
    }
    uint32_t idx = uint32_t(nodes_.size());
    nodes_.push_back(node);
    index_.emplace(key, idx);
}

}  // namespace pa
