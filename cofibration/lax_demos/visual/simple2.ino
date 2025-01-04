// Note: Previous includes and base classes remain the same up to RenderPipeline
// Completing RenderPipeline's triangle rendering implementation first:

void RenderPipeline::renderTriangle(TFT_eSPI& tft, 
                                  const Vertex& v1, 
                                  const Vertex& v2, 
                                  const Vertex& v3,
                                  std::vector<float>& depthBuffer) {
    // Project vertices to screen space
    auto project = [](const Vertex& v) -> Vec2 {
        float scale = 100.0f / (v.z + 3.0f);
        return {
            160.0f + v.x * scale,
            120.0f + v.y * scale
        };
    };

    Vec2 p1 = project(v1);
    Vec2 p2 = project(v2);
    Vec2 p3 = project(v3);

    // Compute bounding box
    int minX = std::max(0, (int)std::min({p1.x, p2.x, p3.x}));
    int minY = std::max(0, (int)std::min({p1.y, p2.y, p3.y}));
    int maxX = std::min(319, (int)std::max({p1.x, p2.x, p3.x}));
    int maxY = std::min(239, (int)std::max({p1.y, p2.y, p3.y}));

    // Edge functions
    auto edge = [](const Vec2& a, const Vec2& b, float x, float y) {
        return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
    };

    // Scan conversion with depth interpolation
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            float w0 = edge(p2, p3, x + 0.5f, y + 0.5f);
            float w1 = edge(p3, p1, x + 0.5f, y + 0.5f);
            float w2 = edge(p1, p2, x + 0.5f, y + 0.5f);

            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                // Compute barycentric coordinates
                float area = edge(p1, p2, p3.x, p3.y);
                w0 /= area;
                w1 /= area;
                w2 /= area;

                // Interpolate depth
                float depth = w0 * v1.z + w1 * v2.z + w2 * v3.z;
                int idx = y * 320 + x;

                if (depth < depthBuffer[idx]) {
                    depthBuffer[idx] = depth;

                    // Color interpolation
                    Color interpolated = Color::interpolate(
                        Color::interpolate(v1.color, v2.color, w1 / (w0 + w1)),
                        v3.color,
                        w2
                    );

                    tft.drawPixel(x, y, interpolated.to565());
                }
            }
        }
    }
}

// Shader implementations
class SchemeShader : public ShaderEffect {
public:
    Color computeColor(float x, float y, float z, float t) override {
        // Create a dynamic color scheme based on local ring structure
        float radius = sqrt(x*x + y*y);
        float angle = atan2(y, x);
        float spectrumValue = sin(5.0f * radius + t) * 0.5f + 0.5f;
        
        return Color{
            static_cast<uint8_t>(255 * sin(angle + t)),
            static_cast<uint8_t>(255 * spectrumValue),
            static_cast<uint8_t>(255 * cos(radius + t))
        };
    }
};

class DerivedCategoryShader : public ShaderEffect {
public:
    Color computeColor(float x, float y, float z, float t) override {
        // Visualize cohomological grading
        float grade = fmod(z + t, 2.0f);
        float complexity = sin(10.0f * x + 8.0f * y + t);
        
        return Color{
            static_cast<uint8_t>(255 * grade),
            static_cast<uint8_t>(255 * (1.0f - grade) * complexity),
            static_cast<uint8_t>(255 * (1.0f - complexity))
        };
    }
};

class ModuliSpaceShader : public ShaderEffect {
public:
    Color computeColor(float x, float y, float z, float t) override {
        // Represent deformation parameters
        float deformation = sin(x * y + t);
        float stability = cos(z * 5.0f + t);
        
        return Color{
            static_cast<uint8_t>(255 * stability),
            static_cast<uint8_t>(255 * deformation),
            static_cast<uint8_t>(255 * (1.0f - stability * deformation))
        };
    }
};

class IntersectionTheoryShader : public ShaderEffect {
public:
    Color computeColor(float x, float y, float z, float t) override {
        // Visualize intersection multiplicities
        float intersection = sin(x * 5.0f) * sin(y * 5.0f);
        float cycle = cos(z * 3.0f + t);
        
        return Color{
            static_cast<uint8_t>(255 * abs(intersection)),
            static_cast<uint8_t>(255 * abs(cycle)),
            static_cast<uint8_t>(255 * abs(intersection * cycle))
        };
    }
};

// Completing visualization methods
void AlgebraicGeometryEngine::Visualizer::visualizeScheme(TFT_eSPI& tft) {
    constexpr int GRID_SIZE = 20;
    
    // Generate scheme structure
    for (int i = -GRID_SIZE; i <= GRID_SIZE; i++) {
        for (int j = -GRID_SIZE; j <= GRID_SIZE; j++) {
            float x = i / float(GRID_SIZE);
            float y = j / float(GRID_SIZE);
            float z = sin(5.0f * x + 4.0f * y + time);
            
            Color color = currentEffect->computeColor(x, y, z, time);
            pipeline.addVertex(x, y, z, color);
            
            // Create triangles
            if (i < GRID_SIZE && j < GRID_SIZE) {
                int base = (i + GRID_SIZE) * (2 * GRID_SIZE + 1) + (j + GRID_SIZE);
                pipeline.addTriangle(base, base + 1, base + 2 * GRID_SIZE + 1);
                pipeline.addTriangle(base + 1, base + 2 * GRID_SIZE + 2, base + 2 * GRID_SIZE + 1);
            }
        }
    }
    
    pipeline.render(tft);
}

void AlgebraicGeometryEngine::Visualizer::visualizeDerivedCategory(TFT_eSPI& tft) {
    // Create complex visualization
    constexpr int NUM_OBJECTS = 10;
    constexpr float RADIUS = 0.8f;
    
    for (int i = 0; i < NUM_OBJECTS; i++) {
        float angle = 2.0f * PI * i / NUM_OBJECTS + time;
        float x = cos(angle) * RADIUS;
        float y = sin(angle) * RADIUS;
        float z = sin(angle * 2.0f + time);
        
        Color color = currentEffect->computeColor(x, y, z, time);
        pipeline.addVertex(x, y, z, color);
        
        // Connect objects with morphisms
        if (i > 0) {
            pipeline.addTriangle(i-1, i, NUM_OBJECTS);
        }
    }
    
    // Center vertex
    pipeline.addVertex(0, 0, 0, Color{255, 255, 255});
    
    pipeline.render(tft);
}

void AlgebraicGeometryEngine::Visualizer::visualizeModuliSpace(TFT_eSPI& tft) {
    constexpr int GRID_SIZE = 15;
    
    // Generate moduli space structure
    for (int i = -GRID_SIZE; i <= GRID_SIZE; i++) {
        for (int j = -GRID_SIZE; j <= GRID_SIZE; j++) {
            float x = i / float(GRID_SIZE);
            float y = j / float(GRID_SIZE);
            
            // Deformation parameters
            float z = 0.2f * sin(10.0f * x + 8.0f * y + time);
            
            Color color = currentEffect->computeColor(x, y, z, time);
            pipeline.addVertex(x, y, z, color);
            
            // Create deformation space
            if (i < GRID_SIZE && j < GRID_SIZE) {
                int base = (i + GRID_SIZE) * (2 * GRID_SIZE + 1) + (j + GRID_SIZE);
                pipeline.addTriangle(base, base + 1, base + 2 * GRID_SIZE + 1);
                pipeline.addTriangle(base + 1, base + 2 * GRID_SIZE + 2, base + 2 * GRID_SIZE + 1);
            }
        }
    }
    
    pipeline.render(tft);
}

void AlgebraicGeometryEngine::Visualizer::visualizeIntersectionTheory(TFT_eSPI& tft) {
    constexpr int NUM_CYCLES = 12;
    constexpr float RADIUS = 0.7f;
    
    // Generate intersection cycles
    for (int i = 0; i < NUM_CYCLES; i++) {
        float angle1 = 2.0f * PI * i / NUM_CYCLES + time;
        
        for (int j = 0; j < NUM_CYCLES; j++) {
            float angle2 = 2.0f * PI * j / NUM_CYCLES - time * 0.5f;
            
            float x = cos(angle1) * RADIUS * cos(angle2);
            float y = sin(angle1) * RADIUS;
            float z = cos(angle1) * RADIUS * sin(angle2);
            
            Color color = currentEffect->computeColor(x, y, z, time);
            pipeline.addVertex(x, y, z, color);
            
            // Create intersection lattice
            if (i < NUM_CYCLES-1 && j < NUM_CYCLES-1) {
                int base = i * NUM_CYCLES + j;
                pipeline.addTriangle(base, base + 1, base + NUM_CYCLES);
                pipeline.addTriangle(base + 1, base + NUM_CYCLES + 1, base + NUM_CYCLES);
            }
        }
    }
    
    pipeline.render(tft);
}

// Additional helper structs
struct Vec2 {
    float x, y;
};

struct Vec3 {
    float x, y, z;
};

struct Vec4 {
    float x, y, z, w;
};

// Main loop handlers remain the same

void setup() {
    visualizer.setup(tft);
}

void loop() {
    visualizer.update(tft);
    delay(16);
}
