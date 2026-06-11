// --- Right Kan extension: local observations -> global inference ------------

struct Observation {
    unsigned n;          // degree
    std::string prefix;  // observed output bits, e.g. "10011"
};

struct Evidence {
    std::string chart;
    std::vector<u64> candidates;
};

using Recognizer = std::function<std::optional<Evidence>(const Observation&)>;

static std::string reverse_bits_string(const std::string& s) {
    return std::string(s.rbegin(), s.rend());
}

static bool matches_prefix(u64 poly, unsigned n, u64 seed, const std::string& prefix, bool reciprocal = false) {
    LFSR lfsr{n, reciprocal ? reciprocal_poly(poly) : poly, seed};
    for (char c : prefix) {
        char bit = lfsr.step() ? '1' : '0';
        if (bit != c) return false;
    }
    return true;
}

static std::vector<u64> intersect_candidate_sets(const std::vector<std::vector<u64>>& sets) {
    if (sets.empty()) return {};

    std::vector<u64> acc = sets.front();
    std::sort(acc.begin(), acc.end());

    for (std::size_t i = 1; i < sets.size(); ++i) {
        std::vector<u64> cur = sets[i];
        std::sort(cur.begin(), cur.end());

        std::vector<u64> next;
        std::set_intersection(
            acc.begin(), acc.end(),
            cur.begin(), cur.end(),
            std::back_inserter(next)
        );
        acc = std::move(next);
    }
    return acc;
}

static Recognizer direct_recognizer = [](const Observation& obs) -> std::optional<Evidence> {
    std::vector<u64> cand;
    for (u64 p : primitive_locus(obs.n)) {
        if (matches_prefix(p, obs.n, 1, obs.prefix, false)) {
            cand.push_back(p);
        }
    }
    return Evidence{"direct chart", std::move(cand)};
};

static Recognizer reciprocal_recognizer = [](const Observation& obs) -> std::optional<Evidence> {
    std::vector<u64> cand;
    std::string rev = reverse_bits_string(obs.prefix);

    for (u64 p : primitive_locus(obs.n)) {
        if (matches_prefix(p, obs.n, 1, rev, true)) {
            cand.push_back(p);
        }
    }
    return Evidence{"reciprocal chart", std::move(cand)};
};

struct GlobalInference {
    std::vector<u64> candidates;
};

static GlobalInference ran_extend(const std::vector<Recognizer>& recognizers, const Observation& obs) {
    std::vector<std::vector<u64>> all_sets;

    for (const auto& rec : recognizers) {
        auto ev = rec(obs);
        if (!ev) continue;
        all_sets.push_back(std::move(ev->candidates));
    }

    return GlobalInference{intersect_candidate_sets(all_sets)};
}

int main() {
    try {
        const unsigned n = 5;
        auto prims = primitive_locus(n);

        if (prims.empty()) {
            std::cout << "No primitive polynomials found.\n";
            return 0;
        }

        u64 target = prims.front();
        std::cout << "Target polynomial: " << poly_to_string(target) << "\n";

        // Generate an observed prefix from the target.
        LFSR probe{n, target, 1};
        std::string prefix;
        for (int i = 0; i < 12; ++i) {
            prefix.push_back(probe.step() ? '1' : '0');
        }

        Observation obs{n, prefix};
        std::cout << "Observed prefix: " << obs.prefix << "\n";

        std::vector<Recognizer> recognizers{
            direct_recognizer,
            reciprocal_recognizer
        };

        auto inf = ran_extend(recognizers, obs);

        std::cout << "\nRight Kan extension result (global candidates):\n";
        for (u64 p : inf.candidates) {
            std::cout << "  " << poly_to_string(p) << "\n";
        }

        if (inf.candidates.empty()) {
            std::cout << "  <no compatible global object>\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
