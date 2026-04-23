#ifndef MAPPINGS_H
#define MAPPINGS_H
#include <complex>
#include <cmath>
#include <algorithm>
typedef std::complex<double> cd;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace interference {
enum class MappingType { Standard = 0, LogPolar = 1, Mobius = 2, EulerSpace = 3, Reciprocal = 4, Joukowsky = 5 };
inline cd apply_mapping(cd z, MappingType type, double param = 1.0) {
    switch (type) {
        case MappingType::LogPolar: { double r = std::abs(z); double t = std::arg(z); return cd(std::log(std::max(1e-15, r)), t); }
        case MappingType::Mobius: { return (z - cd(0, 1)) / (z + cd(0, 1) + 1e-15); }
        case MappingType::EulerSpace: { return std::exp(cd(0, 1) * M_PI * z / param); }
        case MappingType::Reciprocal: { return 1.0 / (z + 1e-15); }
        case MappingType::Joukowsky: { return z + 1.0 / (z + 1e-15); }
        case MappingType::Standard:
        default: return z;
    }
}
}
#endif
