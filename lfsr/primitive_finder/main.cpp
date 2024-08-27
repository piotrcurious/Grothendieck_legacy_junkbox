#include "PrimitivePolynomial.h"
#include <iostream>
#include <cstdlib> // For std::atoi

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <degree> <x-dimension> <y-dimension>" << std::endl;
        return 1;
    }

    // Extract command-line arguments
    int degree = std::atoi(argv[1]);
    int x = std::atoi(argv[2]);
    int y = std::atoi(argv[3]);

    // Check if arguments are valid
    if (degree <= 0 || x <= 0 || y <= 0) {
        std::cerr << "All arguments must be positive integers." << std::endl;
        return 1;
    }

    // Create a PrimitivePolynomial object with the given degree
    PrimitivePolynomial polyFinder(degree);

    // Find and print primitive polynomials of the given degree
    std::vector<std::vector<int>> primitives = polyFinder.findPrimitivePolynomials();
    std::cout << "Primitive polynomials of degree " << degree << ":" << std::endl;
    for (const auto& poly : primitives) {
        polyFinder.printPolynomial(poly);
    }

    // Calculate the required degree for covering the area defined by x and y dimensions
    int neededDegree = std::ceil(std::log2(x * y));
    std::cout << "\nRequired degree to cover an area of " << x << "x" << y << ": " << neededDegree << std::endl;

    // Create a PrimitivePolynomial object for the calculated degree
    PrimitivePolynomial areaPolyFinder(neededDegree);
    std::vector<std::vector<int>> areaPrimitives = areaPolyFinder.findPrimitivePolynomials();

    std::cout << "\nPrimitive polynomials to cover area of " << x << "x" << y << ":\n";
    for (const auto& poly : areaPrimitives) {
        areaPolyFinder.printPolynomial(poly);
    }

    return 0;
}
