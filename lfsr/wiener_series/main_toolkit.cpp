#include "lfsr_wiener_toolkit.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace lfsr_wiener;

int main(int argc, char* argv[]) {
    uint32_t L = 3;
    uint32_t poly = 11; // t^3 + t + 1
    uint32_t beta = 1;

    // Pair synthesis parameters
    bool synthesize = false;
    uint32_t L_B = 4;
    uint32_t poly_B = 19; // t^4 + t + 1
    uint32_t beta_B = 1;
    PairCombinationMode mode = PairCombinationMode::MULTIPLICATIVE;

    std::string outfile = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-L" && i + 1 < argc) L = std::stoul(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) poly = std::stoul(argv[++i]);
        else if (arg == "-b" && i + 1 < argc) beta = std::stoul(argv[++i]);
        else if (arg == "--synthesize") synthesize = true;
        else if (arg == "-LB" && i + 1 < argc) L_B = std::stoul(argv[++i]);
        else if (arg == "-pB" && i + 1 < argc) poly_B = std::stoul(argv[++i]);
        else if (arg == "-bB" && i + 1 < argc) beta_B = std::stoul(argv[++i]);
        else if (arg == "-m" && i + 1 < argc) {
            std::string m_str = argv[++i];
            if (m_str == "additive") mode = PairCombinationMode::ADDITIVE;
            else if (m_str == "multiplexed") mode = PairCombinationMode::MULTIPLEXED;
            else mode = PairCombinationMode::MULTIPLICATIVE;
        }
        else if (arg == "-o" && i + 1 < argc) outfile = argv[++i];
    }

    GF2Field field_A(L, poly);

    std::string json_str;
    if (synthesize) {
        GF2Field field_B(L_B, poly_B);
        PairSynthesisReport pair_report = LFSRSynthesisEngine::synthesize_pair(field_A, beta, field_B, beta_B, mode);
        json_str = LFSRSynthesisEngine::export_pair_json(pair_report);
    } else {
        SpectralReport report = SpectralAnalyzer::analyze(field_A, beta);
        json_str = SpectralAnalyzer::export_json(report);
    }

    if (!outfile.empty()) {
        std::ofstream ofs(outfile);
        ofs << json_str;
        std::cout << "Report written to " << outfile << std::endl;
    } else {
        std::cout << json_str;
    }

    return 0;
}
