#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <vector>
#include <array>
#include <complex>
#include <memory>
#include <cmath>
#include <iostream>
#include <random>

// Constants
const float PI = 3.14159265359f;

// -------------------------------------------------------------------------
// Part 1: Mathematical Foundations
// -------------------------------------------------------------------------

namespace math {

    // Abstract Algebraic Structure
    template<typename T>
    class Field {
    public:
        virtual T add(const T& a, const T& b) const = 0;
        virtual T mul(const T& a, const T& b) const = 0;
        virtual T inv(const T& a) const = 0;
        virtual ~Field() = default;
    };

    // Concrete Field: Complex Numbers
    class ComplexField : public Field<std::complex<float>> {
    public:
        std::complex<float> add(const std::complex<float>& a, const std::complex<float>& b) const override { return a + b; }
        std::complex<float> mul(const std::complex<float>& a, const std::complex<float>& b) const override { return a * b; }
        std::complex<float> inv(const std::complex<float>& a) const override { return 1.0f / a; }
    };

    // Projective Space P^N(C)
    template<size_t N>
    class ProjectiveSpace {
        std::array<std::complex<float>, N+1> coords;
    public:
        ProjectiveSpace(const std::array<std::complex<float>, N+1>& c) : coords(c) {}
        
        // Homogeneous normalization (divide by last non-zero coordinate)
        std::array<std::complex<float>, N+1> normalize() const {
            std::complex<float> norm = 1.0f;
            for (int i = N; i >= 0; --i) {
                if (std::abs(coords[i]) > 1e-6) {
                    norm = coords[i];
                    break;
                }
            }
            std::array<std::complex<float>, N+1> res;
            for(size_t i=0; i<=N; ++i) res[i] = coords[i] / norm;
            return res;
        }

        const std::array<std::complex<float>, N+1>& getCoords() const { return coords; }
    };

    // 3D Vector for Geometry
    struct Vec3 {
        float x, y, z;
        Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
        Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
        
        static Vec3 cross(const Vec3& a, const Vec3& b) {
            return {
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x
            };
        }
        
        void normalize() {
            float len = std::sqrt(x*x + y*y + z*z);
            if(len > 0) { x/=len; y/=len; z/=len; }
        }
    };
}

// -------------------------------------------------------------------------
// Part 2: OpenGL Engine Wrapper
// -------------------------------------------------------------------------

namespace gfx {

    struct Color {
        float r, g, b, a;
    };

    struct Vertex {
        math::Vec3 pos;
        math::Vec3 normal;
        Color color;
    };

    class Mesh {
    public:
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        void clear() {
            vertices.clear();
            indices.clear();
        }

        void addTriangle(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
            unsigned int idx = vertices.size();
            vertices.push_back(v1);
            vertices.push_back(v2);
            vertices.push_back(v3);
            indices.push_back(idx);
            indices.push_back(idx + 1);
            indices.push_back(idx + 2);
            
            // Recalculate face normal for flat shading look (optional)
            math::Vec3 u = {v2.pos.x - v1.pos.x, v2.pos.y - v1.pos.y, v2.pos.z - v1.pos.z};
            math::Vec3 v = {v3.pos.x - v1.pos.x, v3.pos.y - v1.pos.y, v3.pos.z - v1.pos.z};
            math::Vec3 n = math::Vec3::cross(u, v);
            n.normalize();

            // Apply calculated normal to vertices (simple approach)
            vertices[idx].normal = n;
            vertices[idx+1].normal = n;
            vertices[idx+2].normal = n;
        }
    };
}

// -------------------------------------------------------------------------
// Part 3: Visualization Logic (The "Engine")
// -------------------------------------------------------------------------

class AlgebraicVizWindow : public Fl_Gl_Window {
    float time;
    gfx::Mesh currentMesh;
    int currentState; // 0=Scheme, 1=DerivedCat, 2=Moduli, 3=Intersection

public:
    AlgebraicVizWindow(int x, int y, int w, int h, const char *l) 
        : Fl_Gl_Window(x, y, w, h, l), time(0), currentState(0) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_MULTISAMPLE);
    }

    void update() {
        time += 0.02f;
        
        // State transition every ~10 seconds
        if (fmod(time, 10.0f) < 0.02f) {
            currentState = (currentState + 1) % 4;
            std::cout << "Switching to state: " << currentState << std::endl;
        }
        
        generateGeometry();
        redraw();
    }

private:
    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_LIGHTING);
            glEnable(GL_LIGHT0);
            glEnable(GL_COLOR_MATERIAL);
            glEnable(GL_NORMALIZE);
            
            // Basic Light
            GLfloat light_pos[] = { 10.0, 10.0, 10.0, 1.0 };
            glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
            
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluPerspective(45.0, (float)w()/(float)h(), 0.1, 100.0);
            glMatrixMode(GL_MODELVIEW);
        }

        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glLoadIdentity();
        gluLookAt(0, 0, 8, 0, 0, 0, 0, 1, 0);

        // Global Rotation
        glRotatef(time * 10.0f, 0, 1, 0);
        glRotatef(time * 5.0f, 1, 0, 0);

        // Render Mesh
        glBegin(GL_TRIANGLES);
        for (unsigned int i : currentMesh.indices) {
            const auto& v = currentMesh.vertices[i];
            glColor4f(v.color.r, v.color.g, v.color.b, v.color.a);
            glNormal3f(v.normal.x, v.normal.y, v.normal.z);
            glVertex3f(v.pos.x, v.pos.y, v.pos.z);
        }
        glEnd();
    }

    void generateGeometry() {
        currentMesh.clear();
        switch(currentState) {
            case 0: generateScheme(); break;
            case 1: generateDerivedCategory(); break;
            case 2: generateModuliSpace(); break;
            case 3: generateIntersectionTheory(); break;
        }
    }

    // 1. SCHEME: Visualizing Spec(R) as a fiber bundle
    // We render a torus (elliptic curve) deformed by noise
    void generateScheme() {
        const int segs = 50;
        const int rings = 30;
        const float R = 2.0f; // Major radius
        const float r = 0.8f; // Minor radius

        for(int i = 0; i < rings; ++i) {
            for(int j = 0; j < segs; ++j) {
                float theta1 = (float)i / rings * 2 * PI;
                float phi1   = (float)j / segs * 2 * PI;
                float theta2 = (float)(i+1) / rings * 2 * PI;
                float phi2   = (float)(j+1) / segs * 2 * PI;

                auto getPos = [&](float t, float p) -> math::Vec3 {
                    // Deform the ring based on 'local ring' properties (simulated by sine waves)
                    float def = 0.2f * sin(5*t + 3*p + time); 
                    float rr = r + def;
                    return {
                        (R + rr * cos(p)) * cos(t),
                        (R + rr * cos(p)) * sin(t),
                        rr * sin(p)
                    };
                };

                auto getColor = [&](float t, float p) -> gfx::Color {
                    return {0.2f + 0.5f*cos(t), 0.5f + 0.5f*sin(p+time), 0.8f, 1.0f};
                };

                math::Vec3 v1 = getPos(theta1, phi1);
                math::Vec3 v2 = getPos(theta2, phi1);
                math::Vec3 v3 = getPos(theta1, phi2);
                math::Vec3 v4 = getPos(theta2, phi2);

                gfx::Color c = getColor(theta1, phi1);
                
                // Two triangles per quad
                gfx::Vertex vert1 = {v1, {0,0,0}, c};
                gfx::Vertex vert2 = {v2, {0,0,0}, c};
                gfx::Vertex vert3 = {v3, {0,0,0}, c};
                gfx::Vertex vert4 = {v4, {0,0,0}, c};

                currentMesh.addTriangle(vert1, vert2, vert3);
                currentMesh.addTriangle(vert3, vert2, vert4);
            }
        }
    }

    // 2. DERIVED CATEGORY: Simplicial Complex
    // Representing objects in a chain complex as connected tetrahedrons
    void generateDerivedCategory() {
        const int objects = 8;
        for(int i=0; i<objects; ++i) {
            float t = time + i;
            float x = 2.5f * cos(t * 0.5f);
            float y = 2.5f * sin(t * 0.3f);
            float z = 1.0f * sin(t * 1.5f);

            // Create a tetrahedron at (x,y,z)
            math::Vec3 center = {x, y, z};
            float s = 0.4f;

            math::Vec3 p1 = center + math::Vec3{0, s, 0};
            math::Vec3 p2 = center + math::Vec3{-s, -s, s};
            math::Vec3 p3 = center + math::Vec3{s, -s, s};
            math::Vec3 p4 = center + math::Vec3{0, -s, -s};

            gfx::Color col = { (float)i/objects, 1.0f - (float)i/objects, 0.5f, 0.8f };

            auto addTet = [&](math::Vec3 a, math::Vec3 b, math::Vec3 c) {
                currentMesh.addTriangle({a, {0,0,0}, col}, {b, {0,0,0}, col}, {c, {0,0,0}, col});
            };

            addTet(p1, p2, p3);
            addTet(p1, p3, p4);
            addTet(p1, p4, p2);
            addTet(p2, p4, p3);

            // Draw "Morphism" (Line as a thin triangle) to next object
            if (i < objects - 1) {
                // Simplified visualization of morphisms
                // Ideally GL_LINES, but we stick to triangles for uniformity
            }
        }
    }

    // 3. MODULI SPACE: Calabi-Yau Cross Section (Quintic)
    // z1^5 + z2^5 + z3^5 + z4^5 + z5^5 = 0
    // Visualized as a 2D projection into 3D space
    void generateModuliSpace() {
        const int gridSize = 40;
        for(int i=0; i<gridSize; ++i) {
            for(int j=0; j<gridSize; ++j) {
                // Complex plane inputs mapped to surface
                std::complex<float> z1( (i - gridSize/2)*0.1f, (j - gridSize/2)*0.1f );
                
                // Visualize the "potential" of the manifold
                float val = std::abs(std::pow(z1, 5) + std::complex<float>(1,0)); 
                float height = std::log(val + 0.1f) * 0.5f;

                float u = (i - gridSize/2) * 0.15f;
                float v = (j - gridSize/2) * 0.15f;

                // Animate the manifold deformation
                float deformation = sin(u*v + time);

                math::Vec3 pos = {u, height + deformation * 0.2f, v};
                
                // Color based on "stability"
                gfx::Color col = {
                    0.5f + 0.5f*sin(height),
                    0.2f,
                    0.5f + 0.5f*cos(height),
                    1.0f
                };

                // Create points (as small triangles for particle effect feel)
                float s = 0.05f;
                currentMesh.addTriangle(
                    {pos, {0,0,0}, col},
                    {pos + math::Vec3{s,0,0}, {0,0,0}, col},
                    {pos + math::Vec3{0,s,0}, {0,0,0}, col}
                );
            }
        }
    }

    // 4. INTERSECTION THEORY: Bezout's Theorem
    // Intersecting surfaces
    void generateIntersectionTheory() {
        // Surface A: Sine wave
        // Surface B: Cosine wave orthogonal
        const int res = 30;
        const float scale = 3.0f;
        
        // Render Surface A (Red)
        for(int i=0; i<res; ++i) {
            for(int j=0; j<res; ++j) {
                float u = (float)i/res * 2*scale - scale;
                float v = (float)j/res * 2*scale - scale;
                
                auto fnA = [&](float x, float z) { return 0.5f * sin(x + time); };
                
                math::Vec3 p1 = {u, fnA(u,v), v};
                math::Vec3 p2 = {u+0.1f, fnA(u+0.1f,v), v};
                math::Vec3 p3 = {u, fnA(u,v+0.1f), v+0.1f};
                
                gfx::Color red = {1.0f, 0.2f, 0.2f, 0.6f}; // Semi-transparent look logic usually requires sorting, ignoring here
                currentMesh.addTriangle({p1,{0,0,0},red}, {p2,{0,0,0},red}, {p3,{0,0,0},red});
            }
        }

        // Render Surface B (Blue)
        for(int i=0; i<res; ++i) {
            for(int j=0; j<res; ++j) {
                float u = (float)i/res * 2*scale - scale;
                float v = (float)j/res * 2*scale - scale;
                
                auto fnB = [&](float x, float z) { return 0.5f * cos(z - time); }; // Orthogonal wave
                
                math::Vec3 p1 = {u, fnB(u,v), v};
                math::Vec3 p2 = {u+0.1f, fnB(u+0.1f,v), v};
                math::Vec3 p3 = {u, fnB(u,v+0.1f), v+0.1f};
                
                gfx::Color blue = {0.2f, 0.2f, 1.0f, 0.6f};
                currentMesh.addTriangle({p1,{0,0,0},blue}, {p2,{0,0,0},blue}, {p3,{0,0,0},blue});

                // VISUALIZE INTERSECTION LOCUS
                // If |fnA - fnB| is small, draw a bright white particle
                float diff = std::abs(fnB(u,v) - (0.5f * sin(u + time)));
                if(diff < 0.1f) {
                    gfx::Color white = {1.0f, 1.0f, 1.0f, 1.0f};
                    math::Vec3 intersectPos = {u, fnB(u,v), v};
                    float s = 0.08f;
                    currentMesh.addTriangle(
                        {intersectPos + math::Vec3{-s,-s,0}, {0,0,0}, white},
                        {intersectPos + math::Vec3{s,-s,0}, {0,0,0}, white},
                        {intersectPos + math::Vec3{0,s,0}, {0,0,0}, white}
                    );
                }
            }
        }
    }
};

// -------------------------------------------------------------------------
// Main Entry
// -------------------------------------------------------------------------

void timer_cb(void* v) {
    AlgebraicVizWindow* win = (AlgebraicVizWindow*)v;
    win->update();
    Fl::repeat_timeout(1.0/60.0, timer_cb, v);
}

int main(int argc, char** argv) {
    Fl_Window* mainWin = new Fl_Window(800, 600, "Algebraic Geometry Engine (OpenGL)");
    AlgebraicVizWindow* glWin = new AlgebraicVizWindow(10, 10, 780, 580, "Viewport");
    
    mainWin->resizable(glWin);
    mainWin->end();
    mainWin->show(argc, argv);
    
    Fl::add_timeout(1.0/60.0, timer_cb, glWin);
    
    return Fl::run();
}
