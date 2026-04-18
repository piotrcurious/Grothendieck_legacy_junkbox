#include <cmath>
#include <vector>
#include <numeric>
#include <array>
#include <optional>

// Previous FredholmErrorCompensator and AngleQuantizationTracker implementations remain...

// Add these new classes for graphics-specific handling:

template<typename T>
struct Vector2D {
    T x, y;
    
    Vector2D(T x = 0, T y = 0) : x(x), y(y) {}
    
    T magnitude() const {
        return std::sqrt(x * x + y * y);
    }
    
    Vector2D normalized() const {
        T mag = magnitude();
        if (mag == 0) return *this;
        return Vector2D(x / mag, y / mag);
    }
};

template<typename T>
struct Transform2D {
    Vector2D<T> position;
    T rotation;
    Vector2D<T> scale;
    
    Transform2D() : position(), rotation(0), scale(1, 1) {}
};

template<typename T>
class GraphicsQuantizationHandler {
private:
    AngleQuantizationTracker<T> angleTracker;
    static constexpr T PI = T(3.14159265358979323846);
    static constexpr size_t INTERPOLATION_STEPS = 60;  // For 60 FPS animation
    
    struct InterpolationCache {
        std::vector<Transform2D<T>> cachedSteps;
        T totalError;
    };
    
    std::optional<InterpolationCache> cache;
    
    Vector2D<T> rotateVector(const Vector2D<T>& vec, T angle) {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Vector2D<T>(
            vec.x * cosA - vec.y * sinA,
            vec.x * sinA + vec.y * cosA
        );
    }

public:
    // Handle rotation with error compensation for rendering
    Vector2D<T> applyRotation(const Vector2D<T>& point, T angle) {
        auto result = angleTracker.convertToAngle(angle / PI);
        return rotateVector(point, result.angle);
    }
    
    // Smooth interpolation between transforms with error compensation
    Transform2D<T> interpolateTransforms(
        const Transform2D<T>& start,
        const Transform2D<T>& end,
        T t,
        bool useCache = true
    ) {
        // Check if we can use cached interpolation
        if (useCache && cache && t >= 0 && t <= 1) {
            size_t index = static_cast<size_t>(t * (INTERPOLATION_STEPS - 1));
            return cache->cachedSteps[index];
        }
        
        // Calculate new interpolation
        Transform2D<T> result;
        
        // Position interpolation (linear)
        result.position.x = start.position.x + (end.position.x - start.position.x) * t;
        result.position.y = start.position.y + (end.position.y - start.position.y) * t;
        
        // Rotation interpolation (with error compensation)
        T startAngle = angleTracker.convertToAngle(start.rotation / PI).angle;
        T endAngle = angleTracker.convertToAngle(end.rotation / PI).angle;
        
        // Use SLERP for rotation to avoid discontinuities
        T angleDiff = endAngle - startAngle;
        if (angleDiff > PI) angleDiff -= 2 * PI;
        if (angleDiff < -PI) angleDiff += 2 * PI;
        
        result.rotation = startAngle + angleDiff * t;
        
        // Scale interpolation (logarithmic to preserve area ratios)
        result.scale.x = std::exp(std::log(start.scale.x) * (1 - t) + std::log(end.scale.x) * t);
        result.scale.y = std::exp(std::log(start.scale.y) * (1 - t) + std::log(end.scale.y) * t);
        
        return result;
    }
    
    // Pre-calculate interpolation steps for smooth animation
    void precalculateInterpolation(
        const Transform2D<T>& start,
        const Transform2D<T>& end
    ) {
        InterpolationCache newCache;
        newCache.cachedSteps.reserve(INTERPOLATION_STEPS);
        newCache.totalError = 0;
        
        for (size_t i = 0; i < INTERPOLATION_STEPS; ++i) {
            T t = T(i) / (INTERPOLATION_STEPS - 1);
            auto transform = interpolateTransforms(start, end, t, false);
            newCache.cachedSteps.push_back(transform);
            
            // Accumulate error for monitoring
            auto angleResult = angleTracker.convertToAngle(transform.rotation / PI);
            newCache.totalError += std::abs(angleResult.errorEstimate);
        }
        
        cache = std::move(newCache);
    }
    
    // Anti-aliasing for rotated lines
    struct AALineSegment {
        Vector2D<T> start, end;
        T thickness;
        T alpha;
    };
    
    AALineSegment calculateAALine(
        const Vector2D<T>& start,
        const Vector2D<T>& end,
        T angle
    ) {
        auto rotatedAngle = angleTracker.convertToAngle(angle / PI);
        
        // Calculate perpendicular vector for thickness
        Vector2D<T> direction = {end.x - start.x, end.y - start.y};
        T length = direction.magnitude();
        direction = direction.normalized();
        
        // Calculate anti-aliasing parameters based on quantization error
        T errorFactor = std::abs(rotatedAngle.errorEstimate);
        T baseThickness = T(1.0);  // Base line thickness in pixels
        T aaThickness = baseThickness + errorFactor * T(0.5);
        
        // Calculate alpha based on sub-pixel position
        T subPixelOffset = rotatedAngle.angle * length - 
                          std::floor(rotatedAngle.angle * length);
        T alpha = T(1.0) - subPixelOffset;
        
        return {
            start,
            end,
            aaThickness,
            alpha
        };
    }
};

// Example usage for graphics applications
void demonstrateGraphicsUsage() {
    GraphicsQuantizationHandler<double> graphics;
    
    // Example 1: Rotating a sprite
    Vector2D<double> spriteCenter(100.0, 100.0);
    double rotationAngle = 45.0;  // degrees
    Vector2D<double> rotatedPoint = graphics.applyRotation(
        spriteCenter,
        rotationAngle * PI / 180.0
    );
    
    // Example 2: Smooth camera rotation animation
    Transform2D<double> startTransform;
    startTransform.position = {0.0, 0.0};
    startTransform.rotation = 0.0;
    
    Transform2D<double> endTransform;
    endTransform.position = {100.0, 100.0};
    endTransform.rotation = 90.0 * PI / 180.0;
    
    // Precalculate smooth interpolation steps
    graphics.precalculateInterpolation(startTransform, endTransform);
    
    // Use in rendering loop (example)
    for (int frame = 0; frame < 60; ++frame) {
        double t = frame / 59.0;
        Transform2D<double> interpolated = graphics.interpolateTransforms(
            startTransform,
            endTransform,
            t
        );
        // Use interpolated transform for rendering...
    }
    
    // Example 3: Anti-aliased line rendering
    Vector2D<double> lineStart(0.0, 0.0);
    Vector2D<double> lineEnd(100.0, 100.0);
    double lineAngle = 30.0 * PI / 180.0;
    
    auto aaLine = graphics.calculateAALine(lineStart, lineEnd, lineAngle);
    
    // Use in rendering (pseudo-code):
    // renderer.drawLine(
    //     aaLine.start,
    //     aaLine.end,
    //     aaLine.thickness,
    //     aaLine.alpha
    // );
}
