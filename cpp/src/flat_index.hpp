// Flat open-addressing hash index uint64 key -> uint32 node index
// (Phase II, Program 7). Replaces std::unordered_map in the MCCFR hot path.
//
// Linear probing over a power-of-two array of (key, value) slots, max load
// 1/2, splitmix64 finalizer as the hash (packed infoset keys are structured
// bitfields; an identity hash would cluster). Keys use only 38 bits, so the
// all-ones key is a safe empty sentinel. No erase is needed. Iteration
// order is unspecified; exports sort keys, so results do not depend on it.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pa {

class FlatIndex {
public:
    static constexpr uint64_t EMPTY = ~uint64_t(0);

    FlatIndex() { rehash_to(16); }

    size_t size() const { return size_; }
    size_t capacity() const { return keys_.size(); }

    // Index of `key`, or -1.
    inline int64_t find(uint64_t key) const {
        size_t mask = keys_.size() - 1;
        for (size_t i = hash(key) & mask;; i = (i + 1) & mask) {
            uint64_t k = keys_[i];
            if (k == key) return int64_t(vals_[i]);
            if (k == EMPTY) return -1;
        }
    }

    size_t count(uint64_t key) const { return find(key) >= 0 ? 1 : 0; }

    uint32_t at(uint64_t key) const {
        int64_t v = find(key);
        if (v < 0) throw std::out_of_range("FlatIndex::at");
        return uint32_t(v);
    }

    // Insert a key that is not present.
    void emplace(uint64_t key, uint32_t value) {
        if (key == EMPTY) throw std::invalid_argument("reserved key");
        if (2 * (size_ + 1) > keys_.size()) rehash_to(keys_.size() * 2);
        insert_unchecked(key, value);
        ++size_;
    }

    void reserve(size_t n) {
        size_t cap = 16;
        while (cap < 2 * n) cap <<= 1;
        if (cap > keys_.size()) rehash_to(cap);
    }

    // Test support: shrink to the smallest legal capacity so later inserts
    // trigger many rehashes.
    void shrink_for_tests() {
        size_t cap = 16;
        while (cap < 2 * size_ + 2) cap <<= 1;
        rehash_to(cap);
    }

    struct const_iterator {
        const FlatIndex* m;
        size_t i;
        void skip() { while (i < m->keys_.size() && m->keys_[i] == EMPTY) ++i; }
        std::pair<uint64_t, uint32_t> operator*() const { return {m->keys_[i], m->vals_[i]}; }
        const_iterator& operator++() { ++i; skip(); return *this; }
        bool operator!=(const const_iterator& o) const { return i != o.i; }
    };
    const_iterator begin() const { const_iterator it{this, 0}; it.skip(); return it; }
    const_iterator end() const { return const_iterator{this, keys_.size()}; }

private:
    static inline uint64_t hash(uint64_t x) {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }
    void insert_unchecked(uint64_t key, uint32_t value) {
        size_t mask = keys_.size() - 1;
        size_t i = hash(key) & mask;
        while (keys_[i] != EMPTY) i = (i + 1) & mask;
        keys_[i] = key;
        vals_[i] = value;
    }
    void rehash_to(size_t cap) {
        std::vector<uint64_t> ok;
        std::vector<uint32_t> ov;
        ok.swap(keys_);
        ov.swap(vals_);
        keys_.assign(cap, EMPTY);
        vals_.assign(cap, 0);
        for (size_t i = 0; i < ok.size(); ++i)
            if (ok[i] != EMPTY) insert_unchecked(ok[i], ov[i]);
    }

    std::vector<uint64_t> keys_;
    std::vector<uint32_t> vals_;
    size_t size_ = 0;
};

}  // namespace pa
