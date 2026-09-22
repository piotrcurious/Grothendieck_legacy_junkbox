#include "lfsr_wiener_toolkit.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace lfsr_wiener;

int main(int argc, char* argv[]) {
    uint32_t L = 3;
    uint32_t poly = 11; // t^3 + t + 1
    uint32_t beta = 1;
    std::string outfile = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-L" && i + 1 < argc) L = std::stoul(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) poly = std::stoul(argv[++i]);
        else if (arg == "-b" && i + 1 < argc) beta = std::stoul(argv[++i]);
        else if (arg == "-o" && i + 1 < argc) outfile = argv[++i];
    }

    GF2Field field(L, poly);
    SpectralReport report = SpectralAnalyzer::analyze(field, beta);

    std::string json_str = SpectralAnalyzer::export_json(report);

    if (!outfile.empty()) {
        std::ofstream ofs(outfile);
        ofs << json_str;
        std::cout << "Report written to " << outfile << std::endl;
    } else {
        std::cout << json_str;
    }

    return 0;
}
