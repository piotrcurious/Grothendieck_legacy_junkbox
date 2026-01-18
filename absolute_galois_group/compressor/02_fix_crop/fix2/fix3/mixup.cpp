#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <Eigen/Dense>
#include <memory>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

using namespace Eigen;
using namespace std;

// --- Architecture Configuration ---
const int POLY_ORDER = 4; // Quartic polynomials (Basis size = 5x5 = 25)
const int DICT_SIZE = 128; // Learned "Meta-Atoms" (Combinations of polynomials)
const int MIN_BLOCK = 4;
const int MAX_BLOCK = 64;
const float VAR_THRESHOLD = 50.0f; 

// --- Mathematical Abstractions ---

// Abstract Kernel for continuous function approximation
class OrthogonalKernel {
public:
    virtual float eval(float x, float y, int basis_idx) const = 0;
    virtual int basis_size() const = 0;
    virtual ~OrthogonalKernel() = default;
};

// Legendre Polynomial Basis (L^2[-1, 1] orthogonal)
class Legendre2DKernel : public OrthogonalKernel {
    int order;
    int dim;
public:
    Legendre2DKernel(int n) : order(n), dim((n+1)*(n+1)) {}

    // Evaluate nth Legendre polynomial at x using recurrence
    float legendre(int n, float x) const {
        if (n == 0) return 1.0f;
        if (n == 1) return x;
        float p0 = 1.0f, p1 = x, p2;
        for (int i = 2; i <= n; ++i) {
            p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i;
            p0 = p1; p1 = p2;
        }
        return p1;
    }

    // Basis index k maps to (i, j) tensor product
    float eval(float x, float y, int k) const override {
        int side = order + 1;
        int i = k / side;
        int j = k % side;
        // Normalized Legendre polynomials: sqrt((2n+1)/2)
        float norm_x = sqrt((2.0f * i + 1.0f) / 2.0f);
        float norm_y = sqrt((2.0f * j + 1.0f) / 2.0f);
        return (legendre(i, x) * norm_x) * (legendre(j, y) * norm_y);
    }

    int basis_size() const override { return dim; }
};

// Global Kernel Instance
unique_ptr<OrthogonalKernel> kernel = make_unique<Legendre2DKernel>(POLY_ORDER);

// --- Data Structures ---

struct CompressedData {
    int orig_width, orig_height;
    int padded_width, padded_height;
    // Dictionary is now coefficients of the Kernel Basis
    // Size: (KernelBasisSize x DICT_SIZE)
    MatrixXf poly_dictionary; 
    std::vector<bool> split_tree;
    std::vector<uint8_t> indices;
    std::vector<uint8_t> scales;
    std::vector<uint8_t> offsets;
    std::vector<uint16_t> block_sizes;
};

// --- Functional Analysis Helpers ---

// "Lebesgue-like" Projection: Continuous projection onto the discrete grid
// Projects an image block onto the Polynomial Basis
VectorXf project_onto_basis(const MatrixXf& block) {
    int h = block.rows();
    int w = block.cols();
    int b_size = kernel->basis_size();
    VectorXf coeffs = VectorXf::Zero(b_size);

    // Gauss-Legendre Quadrature-like sum over the discrete grid
    // Mapping discrete [0, h-1] to continuous [-1, 1]
    for (int r = 0; r < h; ++r) {
        float y = -1.0f + 2.0f * (r + 0.5f) / h;
        for (int c = 0; c < w; ++c) {
            float x = -1.0f + 2.0f * (c + 0.5f) / w;
            float val = block(r, c);
            
            for (int k = 0; k < b_size; ++k) {
                coeffs(k) += val * kernel->eval(x, y, k);
            }
        }
    }
    // Normalization factor (Riemann sum approximation of integral)
    coeffs *= (4.0f / (w * h)); 
    return coeffs;
}

// Reconstruct image block from Basis Coefficients
MatrixXf reconstruction_map(const VectorXf& coeffs, int h, int w) {
    MatrixXf out(h, w);
    int b_size = kernel->basis_size();
    
    // We can precompute the basis grid for fixed sizes to speed this up, 
    // but calculation on the fly proves "resolution independence".
    for (int r = 0; r < h; ++r) {
        float y = -1.0f + 2.0f * (r + 0.5f) / h;
        for (int c = 0; c < w; ++c) {
            float x = -1.0f + 2.0f * (c + 0.5f) / w;
            float sum = 0.0f;
            for (int k = 0; k < b_size; ++k) {
                sum += coeffs(k) * kernel->eval(x, y, k);
            }
            out(r, c) = sum;
        }
    }
    return out;
}

struct Image { int width, height; MatrixXf data; };

Image loadPGM(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("File error");
    string m; file >> m;
    int w, h, mv; file >> w >> h >> mv; file.ignore(1);
    Image img{w, h, MatrixXf(h, w)};
    vector<uint8_t> buf(w*h);
    file.read((char*)buf.data(), buf.size());
    for(int i=0; i<h*w; ++i) img.data(i/w, i%w) = buf[i];
    return img;
}

void savePGM(const string& filename, const Image& img) {
    ofstream file(filename, ios::binary);
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    vector<uint8_t> buf(img.width * img.height);
    for(int i=0; i<img.height; ++i)
        for(int j=0; j<img.width; ++j)
            buf[i*img.width+j] = (uint8_t)clamp(img.data(i,j), 0.f, 255.f);
    file.write((char*)buf.data(), buf.size());
}

// --- Hybrid Learning: Learn Sparse Combinations of Polynomials ---
MatrixXf learn_polynomial_dictionary(const Image& img) {
    cout << "[*] Learning Polynomial Manifold..." << endl;
    vector<VectorXf> samples;
    
    // 1. Gather polynomial projections of random blocks
    // This maps pixel space -> coefficient space (Dimension reduction if block > 5x5)
    mt19937 rng(12345);
    int steps[] = {8, 16, 32};
    
    for (int s : steps) {
        for (int i=0; i+s <= img.height; i+=s/2) {
            for (int j=0; j+s <= img.width; j+=s/2) {
                if (rng()%5 != 0) continue;
                MatrixXf blk = img.data.block(i, j, s, s);
                // Remove DC component (Stored separately as offset)
                float mean = blk.mean();
                blk.array() -= mean;
                
                // Project onto Legendre Basis
                VectorXf coeffs = project_onto_basis(blk);
                
                // Normalize in Coefficient Space (Unit Energy functions)
                float n = coeffs.norm();
                if (n > 1.0f) {
                    coeffs /= n;
                    samples.push_back(coeffs);
                }
            }
        }
    }
    
    // 2. K-SVD / K-Means on the Coefficient Space
    int dim = kernel->basis_size();
    MatrixXf dict = MatrixXf::Random(dim, DICT_SIZE);
    
    // Normalize initial dictionary atoms
    for(int k=0; k<DICT_SIZE; ++k) dict.col(k).normalize();
    
    cout << "    Clustering " << samples.size() << " polynomial vectors..." << endl;

    for (int iter=0; iter<15; ++iter) {
        MatrixXf centroids = MatrixXf::Zero(dim, DICT_SIZE);
        vector<int> counts(DICT_SIZE, 0);
        
        #pragma omp parallel for
        for (size_t i=0; i<samples.size(); ++i) {
            VectorXf::Index best;
            (dict.transpose() * samples[i]).maxCoeff(&best);
            #pragma omp critical
            {
                centroids.col(best) += samples[i];
                counts[best]++;
            }
        }
        
        float shift = 0;
        for (int k=0; k<DICT_SIZE; ++k) {
            if (counts[k] > 0) {
                centroids.col(k) /= counts[k];
                centroids.col(k).normalize();
                shift += (centroids.col(k) - dict.col(k)).norm();
                dict.col(k) = centroids.col(k);
            } else {
                 dict.col(k) = samples[rng() % samples.size()]; // Re-init
            }
        }
        if (shift < 1e-3) break;
    }
    
    return dict;
}

void process_quadrant(const Eigen::Ref<const MatrixXf>& img_block, const MatrixXf& dict, CompressedData& storage) {
    int h = img_block.rows(); int w = img_block.cols();
    float mean = img_block.mean();
    float var = (img_block.array() - mean).square().mean();
    
    bool can_split = (h >= 2*MIN_BLOCK && w >= 2*MIN_BLOCK);
    bool must_split = (h > MAX_BLOCK || w > MAX_BLOCK);
    
    // Variance threshold logic
    if (must_split || (can_split && var > VAR_THRESHOLD)) {
        storage.split_tree.push_back(true);
        int hh = h/2, hw = w/2;
        process_quadrant(img_block.block(0,0,hh,hw), dict, storage);
        process_quadrant(img_block.block(0,hw,hh,w-hw), dict, storage);
        process_quadrant(img_block.block(hh,0,h-hh,hw), dict, storage);
        process_quadrant(img_block.block(hh,hw,h-hh,w-hw), dict, storage);
    } else {
        storage.split_tree.push_back(false);
        
        // 1. Remove DC
        MatrixXf ac_block = img_block.array() - mean;
        
        // 2. Project raw pixels onto Legendre Basis
        VectorXf coeffs = project_onto_basis(ac_block);
        
        // 3. Find matching atom in Polynomial Coefficient Dictionary
        // This is "Sparse Coding" in the Legendre domain
        float norm = coeffs.norm();
        if (norm < 1e-4) norm = 1.0f; // Silence division by zero
        VectorXf normalized_coeffs = coeffs / norm;
        
        VectorXf::Index best_k;
        float alpha = (dict.transpose() * normalized_coeffs).maxCoeff(&best_k);
        
        // Recover scale: Real scale = norm * alpha
        // But since we store normalized atom, the reconstructed coeff vector is:
        // atom * (norm * alpha). Note that alpha is cos(theta). 
        // Best approximation magnitude is dot product magnitude.
        float final_scale = (dict.col(best_k).dot(coeffs));
        
        storage.indices.push_back((uint8_t)best_k);
        storage.scales.push_back((uint8_t)min(255.f, max(0.f, final_scale * 0.5f))); // Scale
        storage.offsets.push_back((uint8_t)min(255.f, max(0.f, mean)));
        storage.block_sizes.push_back((uint16_t)((h << 8) | w));
    }
}

void compress(const string& fin, const string& fout) {
    Image img = loadPGM(fin);
    int oh = img.height, ow = img.width;
    int ph = 1, pw = 1;
    while(ph < oh) ph*=2; while(pw < ow) pw*=2;
    
    MatrixXf data = MatrixXf::Zero(ph, pw);
    data.block(0,0,oh,ow) = img.data;
    
    MatrixXf dict = learn_polynomial_dictionary(img);
    CompressedData cd;
    cd.orig_height = oh; cd.orig_width = ow;
    cd.padded_height = ph; cd.padded_width = pw;
    cd.poly_dictionary = dict;
    
    process_quadrant(data, dict, cd);
    
    ofstream file(fout, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor());
    out.push(file);
    ostream os(&out);
    
    os.write((char*)&oh, 4); os.write((char*)&ow, 4);
    os.write((char*)&ph, 4); os.write((char*)&pw, 4);
    os.write((char*)dict.data(), dict.size()*sizeof(float));
    
    int ts = cd.split_tree.size();
    os.write((char*)&ts, 4);
    vector<uint8_t> pt((ts+7)/8, 0);
    for(int i=0; i<ts; ++i) if(cd.split_tree[i]) pt[i/8] |= (1<<(i%8));
    os.write((char*)pt.data(), pt.size());
    
    int lc = cd.indices.size();
    os.write((char*)&lc, 4);
    os.write((char*)cd.indices.data(), lc);
    os.write((char*)cd.scales.data(), lc);
    os.write((char*)cd.offsets.data(), lc);
    os.write((char*)cd.block_sizes.data(), lc*2);
}

struct Reader {
    vector<bool> tree;
    vector<uint8_t> idx, sc, off;
    vector<uint16_t> sz;
    int ti=0, li=0;
};

void reconstruct(Eigen::Ref<MatrixXf> blk, const MatrixXf& dict, Reader& r) {
    if(r.ti >= r.tree.size()) return;
    if(r.tree[r.ti++]) {
        int h = blk.rows(), w = blk.cols();
        reconstruct(blk.block(0,0,h/2,w/2), dict, r);
        reconstruct(blk.block(0,w/2,h/2,w-w/2), dict, r);
        reconstruct(blk.block(h/2,0,h-h/2,w/2), dict, r);
        reconstruct(blk.block(h/2,w/2,h-h/2,w-w/2), dict, r);
    } else {
        if(r.li >= r.idx.size()) return;
        int k = r.idx[r.li];
        float s = r.sc[r.li] * 2.0f;
        float o = r.off[r.li];
        uint16_t bsz = r.sz[r.li++];
        int bh = (bsz>>8)&0xFF, bw = bsz&0xFF;
        
        // 1. Get coefficients from dictionary
        VectorXf coeffs = dict.col(k);
        
        // 2. Scale
        coeffs *= s;
        
        // 3. Reconstruct continuous function -> discrete grid
        // The Legendre2DKernel class handles the "Lebesgue" evaluation
        MatrixXf res = reconstruction_map(coeffs, bh, bw);
        
        blk = res.array() + o;
    }
}

void decompress(const string& fin, const string& fout) {
    ifstream file(fin, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::zlib_decompressor());
    in.push(file);
    istream is(&in);
    
    int oh, ow, ph, pw;
    is.read((char*)&oh, 4); is.read((char*)&ow, 4);
    is.read((char*)&ph, 4); is.read((char*)&pw, 4);
    
    MatrixXf dict(kernel->basis_size(), DICT_SIZE);
    is.read((char*)dict.data(), dict.size()*4);
    
    int ts; is.read((char*)&ts, 4);
    vector<uint8_t> pt((ts+7)/8);
    is.read((char*)pt.data(), pt.size());
    Reader r; r.tree.resize(ts);
    for(int i=0; i<ts; ++i) r.tree[i] = (pt[i/8] >> (i%8)) & 1;
    
    int lc; is.read((char*)&lc, 4);
    r.idx.resize(lc); r.sc.resize(lc); r.off.resize(lc); r.sz.resize(lc);
    is.read((char*)r.idx.data(), lc);
    is.read((char*)r.sc.data(), lc);
    is.read((char*)r.off.data(), lc);
    is.read((char*)r.sz.data(), lc*2);
    
    MatrixXf data = MatrixXf::Zero(ph, pw);
    reconstruct(data, dict, r);
    
    Image img{ow, oh, data.block(0,0,oh,ow)};
    savePGM(fout, img);
}

int main(int argc, char** argv) {
    if(argc<4) return 1;
    string m = argv[1];
    if(m=="c") compress(argv[2], argv[3]);
    else if(m=="d") decompress(argv[2], argv[3]);
    return 0;
}
