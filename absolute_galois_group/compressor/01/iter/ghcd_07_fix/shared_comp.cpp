#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <cstdint>
#include <cfloat>
#include <Eigen/Dense>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

/**
 * Galois-DCT Hybrid Compressor (GDHC) - v5.4 "Entropy Optimization"
 * Fixed spatial_entries gathering logic to ensure dictionary density.
 */

using namespace Eigen;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum Morphism : uint8_t {
    ID = 0, ROT90, ROT180, ROT270, FLIP_H, FLIP_V, TRANS, ANTITRANS,
    PHASE_H, PHASE_V, PHASE_D, EXP_1, EXP_2, LOG_1, SQUASH, STRETCH,
    NEG_BIT = 16
};

// --- Helpers for Eigen transformations ---
static MatrixXf eval_row_rev(const MatrixXf &m) { MatrixXf r = m; for (int i=0;i<m.rows();++i) r.row(i) = m.row(i).reverse(); return r; }
static MatrixXf eval_col_rev(const MatrixXf &m) { MatrixXf r = m; for (int j=0;j<m.cols();++j) r.col(j) = m.col(j).reverse(); return r; }

MatrixXf apply_galois_action(const MatrixXf& block, uint8_t m) {
    MatrixXf res;
    bool negate = (m & NEG_BIT);
    uint8_t op = m & 0x0F;

    if (op < 8) {
        switch(op) {
            case ROT90:     res = block.transpose().colwise().reverse().eval(); break;
            case ROT180:    res = block.reverse().eval(); break;
            case ROT270:    res = block.transpose().rowwise().reverse().eval(); break;
            case FLIP_H:    res = block.rowwise().reverse().eval(); break;
            case FLIP_V:    res = block.colwise().reverse().eval(); break;
            case TRANS:     res = block.transpose().eval(); break;
            case ANTITRANS: res = block.transpose().reverse().eval(); break;
            default:        res = block; break;
        }
    } else {
        res = block;
        switch(op) {
            case PHASE_H: res.rowwise().reverseInPlace(); break;
            case PHASE_V: res.colwise().reverseInPlace(); break;
            case EXP_1:   res = res.unaryExpr([](float v){ return std::exp(std::min(v, 10.0f)); }); break;
            case EXP_2:   res = res.unaryExpr([](float v){ return std::exp(std::tanh(v)); }); break;
            case LOG_1:   res = res.unaryExpr([](float v){ float a = std::abs(v); return (a>1e-12f) ? std::log(a) : 0.0f; }); break;
            case SQUASH:  res = res.unaryExpr([](float v){ return 1.0f / (1.0f + std::exp(-std::clamp(v, -10.0f, 10.0f))); }); break;
            case STRETCH: res = res.unaryExpr([](float v){ return std::tanh(v); }); break;
            default: break;
        }
    }
    if (negate) res = -res;
    return res;
}

MatrixXf apply_spectral_action(const MatrixXf& coeffs, uint8_t m) {
    bool negate = (m & NEG_BIT);
    uint8_t op = m & 0x0F;
    MatrixXf res = coeffs;
    if (op & 1) res = res.transpose().eval();
    if (op & 2) res = eval_row_rev(res);
    if (op & 4) res = eval_col_rev(res);
    if (negate) res = -res;
    return res;
}

struct CompressionConfig {
    int spatial_block = 16;
    int spatial_stride = 8;
    int spatial_entries = 256;
    int spectral_block = 8;
    int spectral_coeffs = 4;
    int spectral_entries = 128;
    float quant_step = 2.0f;
    float dict_min_variance = 0.5f;

    void validate(){
        if (spectral_coeffs > spectral_block) spectral_coeffs = spectral_block;
        if (spatial_stride > spatial_block) spatial_stride = spatial_block;
        if (spatial_block <= 0) spatial_block = 8;
        if (spectral_block <= 0) spectral_block = 8;
        if (spatial_stride <= 0) spatial_stride = spatial_block / 2;
        if (spectral_coeffs <= 0) spectral_coeffs = 1;
        if (spatial_entries <= 0) spatial_entries = 1;
        if (spectral_entries <= 1) spectral_entries = 1;
        if (dict_min_variance < 0) dict_min_variance = 0.1f;
    }
};

struct Image { int width=0, height=0; MatrixXf data; };

#pragma pack(push,1)
struct FieldEntry { uint16_t id; uint8_t morphism; uint8_t luma; uint8_t chroma; };
#pragma pack(pop)

Image loadPGM(const string &filename){
    ifstream f(filename, ios::binary); if(!f){ cerr<<"IO Error: "<<filename<<"\n"; exit(1);} 
    string magic; f>>magic; if(magic!="P5"){ cerr<<"Only P5 PGM supported\n"; exit(1);} 
    auto skip=[&](ifstream &in){ while(isspace(in.peek())) in.get(); while(in.peek()=='#'){ string d; getline(in,d);} };
    skip(f); int w,h,maxv; f>>w; skip(f); f>>h; skip(f); f>>maxv; f.get();
    Image img; img.width=w; img.height=h; img.data.resize(h,w);
    vector<unsigned char> buf((size_t)w*(size_t)h);
    f.read(reinterpret_cast<char*>(buf.data()), buf.size()); if(!f){ cerr<<"PGM read failed\n"; exit(1);} 
    for(int i=0;i<h;++i) for(int j=0;j<w;++j) img.data(i,j)=static_cast<float>(buf[i*w+j]);
    return img;
}

void savePGM(const string &filename, const Image &img){
    ofstream f(filename, ios::binary); f<<"P5\n"<<img.width<<" "<<img.height<<"\n255\n";
    vector<unsigned char> buf((size_t)img.width*(size_t)img.height);
    for(int i=0;i<img.height;++i) for(int j=0;j<img.width;++j){ 
        float v=img.data(i,j); 
        v=std::max(0.0f,std::min(255.0f,v)); 
        buf[i*img.width+j]=static_cast<unsigned char>(std::lround(v)); 
    }
    f.write(reinterpret_cast<char*>(buf.data()), buf.size()); 
}

class ManifoldDictionary {
public:
    vector<MatrixXf> atoms; bool is_spectral;
    ManifoldDictionary(bool spec=false):is_spectral(spec){}

    // Enhanced training logic to ensure spatial_entries are actually filled
    void train_from_image(const MatrixXf &data, int B, int S, int max_entries, float min_var){
        struct Cand{ float v; MatrixXf m; };
        vector<Cand> cand;
        // Use stride S instead of block size B for much higher candidate density
        for(int i=0; i + B <= data.rows(); i += S){
            for(int j=0; j + B <= data.cols(); j += S){
                MatrixXf blk = data.block(i,j,B,B);
                MatrixXf centered = blk.array() - blk.mean();
                float var = centered.norm();
                if(var > min_var) cand.push_back({var, centered/var});
            }
        }
        
        // Sort candidates by variance (importance)
        sort(cand.begin(), cand.end(), [](auto &a, auto &b){ return a.v > b.v; });
        
        // Populate dictionary with unique manifolds
        for(auto &c: cand){ 
            if((int)atoms.size() >= max_entries) break; 
            bool exists = false;
            // The 0.95 threshold is quite strict; lowered to 0.90 to allow more "variation" 
            // but still keep the dictionary compact.
            float similarity_threshold = 0.90f; 

            for(auto &a: atoms){ 
                if(a.rows() != c.m.rows() || a.cols() != c.m.cols()) continue; 
                for(int m=0; m < 32; ++m){ 
                    MatrixXf morp = is_spectral ? apply_spectral_action(a, (uint8_t)m) : apply_galois_action(a, (uint8_t)m);
                    float dot = (morp.array() * c.m.array()).sum(); 
                    if(std::abs(dot) > similarity_threshold){ exists = true; break; } 
                }
                if(exists) break; 
            }
            if(!exists) atoms.push_back(c.m);
        }
        cout << "Dictionary trained: " << atoms.size() << "/" << max_entries << " entries captured." << endl;
    }

    void learn_from_samples(const vector<MatrixXf> &samples, int limit){
        if(samples.empty() || limit <= 0) return;
        struct Node{ float score; MatrixXf m; };
        vector<Node> candidates; candidates.reserve(samples.size());
        for(const auto &s: samples){ float n=s.norm(); if(n > 1e-5f) candidates.push_back({n, s/n}); }
        sort(candidates.begin(), candidates.end(), [](const Node&a, const Node&b){ return a.score > b.score; });
        
        for(const auto &c: candidates){ 
            if((int)atoms.size() >= limit) break; 
            bool exists = false; 
            for(const auto &a: atoms){ 
                if(a.rows() != c.m.rows() || a.cols() != c.m.cols()) continue;
                for(int m=0; m < 32; ++m){ 
                    MatrixXf morp = is_spectral ? apply_spectral_action(a, (uint8_t)m) : apply_galois_action(a, (uint8_t)m);
                    float dot = (morp.array() * c.m.array()).sum(); 
                    if(std::abs(dot) > 0.85f){ exists = true; break; } 
                }
                if(exists) break; 
            }
            if(!exists) atoms.push_back(c.m); 
        }
    }

    FieldEntry solve(const MatrixXf &target, float &scale){
        FieldEntry best = {0, 0, 0, 0}; scale = 0.0f; 
        if(atoms.empty()) return best;
        float tnorm = target.norm(); if(tnorm < 1e-12f) { scale = 0.0f; return best; }
        float best_corr = -2.0f; // Correlations can be negative if NEG_BIT isn't used

        for(size_t i=0; i < atoms.size(); ++i){ 
            const MatrixXf &a = atoms[i]; 
            if(a.rows() != target.rows() || a.cols() != target.cols()) continue;
            for(int m=0; m < 32; ++m){ 
                MatrixXf morp = is_spectral ? apply_spectral_action(a, (uint8_t)m) : apply_galois_action(a, (uint8_t)m);
                float anorm = morp.norm(); if(anorm < 1e-12f) continue; 
                float dot = (morp.array() * target.array()).sum(); 
                float corr = dot / (anorm * tnorm);
                if(corr > best_corr){ 
                    best_corr = corr; 
                    best.id = static_cast<uint16_t>(i); 
                    best.morphism = static_cast<uint8_t>(m); 
                }
            }
        }
        scale = best_corr; 
        return best;
    }
};

void compress(const string &in, const string &out, CompressionConfig cfg){
    cfg.validate(); Image img = loadPGM(in);
    stringstream ss;
    int32_t w32 = static_cast<int32_t>(img.width), h32 = static_cast<int32_t>(img.height);
    int32_t sb = static_cast<int32_t>(cfg.spatial_block), st = static_cast<int32_t>(cfg.spatial_stride);
    int32_t spb = static_cast<int32_t>(cfg.spectral_block), spc = static_cast<int32_t>(cfg.spectral_coeffs);
    
    ss.write(reinterpret_cast<const char*>(&w32), sizeof(int32_t)); 
    ss.write(reinterpret_cast<const char*>(&h32), sizeof(int32_t));
    ss.write(reinterpret_cast<const char*>(&sb), sizeof(int32_t)); 
    ss.write(reinterpret_cast<const char*>(&st), sizeof(int32_t));
    ss.write(reinterpret_cast<const char*>(&spb), sizeof(int32_t)); 
    ss.write(reinterpret_cast<const char*>(&spc), sizeof(int32_t));
    ss.write(reinterpret_cast<const char*>(&cfg.quant_step), sizeof(float));

    // Fix: train_from_image now correctly takes spatial_stride and spatial_entries
    ManifoldDictionary sdict(false);
    sdict.train_from_image(img.data, cfg.spatial_block, cfg.spatial_stride, cfg.spatial_entries, cfg.dict_min_variance);

    MatrixXf recon = MatrixXf::Zero(img.height, img.width); 
    MatrixXf weight = MatrixXf::Zero(img.height, img.width);
    vector<FieldEntry> spatial_entries_vec; 
    spatial_entries_vec.reserve((img.height/cfg.spatial_stride)*(img.width/cfg.spatial_stride));

    for(int i=0; i+cfg.spatial_block <= img.height; i+=cfg.spatial_stride){
        for(int j=0; j+cfg.spatial_block <= img.width; j+=cfg.spatial_stride){ 
            MatrixXf blk = img.data.block(i,j,cfg.spatial_block,cfg.spatial_block);
            float mu = static_cast<float>(blk.mean()); 
            MatrixXf centered = blk.array() - mu; 
            float sigma = centered.norm(); 
            if(sigma > 1e-6f) centered /= sigma;
            
            float scale; 
            FieldEntry e = sdict.solve(centered, scale);
            e.luma = static_cast<uint8_t>(std::clamp(mu, 0.0f, 255.0f)); 
            // Store norm in chroma slot (scaled for 8-bit)
            e.chroma = static_cast<uint8_t>(std::clamp(sigma * 0.5f, 0.0f, 255.0f));
            spatial_entries_vec.push_back(e);
            
            if(e.id < sdict.atoms.size()){ 
                MatrixXf atom = apply_galois_action(sdict.atoms[e.id], e.morphism);
                recon.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += (atom.array() * (static_cast<float>(e.chroma) * 2.0f)) + static_cast<float>(e.luma);
            } else { 
                recon.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += static_cast<float>(e.luma); 
            }
            weight.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += 1.0f;
        }
    }
    recon = recon.array() / weight.array().max(1.0f);

    MatrixXf res = img.data - recon; 
    int sbk = cfg.spectral_block; 
    int sc = cfg.spectral_coeffs; 
    MatrixXf T(sbk,sbk);
    for(int r=0;r<sbk;++r) for(int c=0;c<sbk;++c) T(r,c) = (r==0)? static_cast<float>(1.0/sqrt((double)sbk)) : static_cast<float>(sqrt(2.0/sbk)*cos((2*c+1)*r*M_PI/(2.0*sbk)));

    vector<MatrixXf> spec_samples;
    for(int i=0; i+sbk <= img.height; i += sbk) {
        for(int j=0; j+sbk <= img.width; j += sbk){ 
            MatrixXf fullc = T * res.block(i,j,sbk,sbk) * T.transpose(); 
            spec_samples.push_back(fullc.block(0,0,sc,sc)); 
        }
    }

    ManifoldDictionary sdict_spec(true); 
    sdict_spec.learn_from_samples(spec_samples, cfg.spectral_entries);

    uint32_t s_count = static_cast<uint32_t>(sdict.atoms.size()); 
    ss.write(reinterpret_cast<const char*>(&s_count), sizeof(uint32_t));
    for(const auto &a: sdict.atoms) ss.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.size()*sizeof(float)));
    
    uint32_t sp_count = static_cast<uint32_t>(sdict_spec.atoms.size()); 
    ss.write(reinterpret_cast<const char*>(&sp_count), sizeof(uint32_t));
    for(const auto &a: sdict_spec.atoms) ss.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.size()*sizeof(float)));

    for(const auto &e: spatial_entries_vec) ss.write(reinterpret_cast<const char*>(&e), sizeof(FieldEntry));

    for(int i=0; i+sbk <= img.height; i += sbk){ 
        for(int j=0; j+sbk <= img.width; j += sbk){ 
            MatrixXf fullc = T * res.block(i,j,sbk,sbk) * T.transpose(); 
            float n = 0.0f; 
            FieldEntry e = sdict_spec.solve(fullc.block(0,0,sc,sc), n); 
            // n here is correlation; multiply by target norm for actual scale
            float actual_scale = n * fullc.block(0,0,sc,sc).norm();
            e.chroma = static_cast<uint8_t>(std::clamp(actual_scale / cfg.quant_step, 0.0f, 255.0f)); 
            ss.write(reinterpret_cast<const char*>(&e), sizeof(FieldEntry)); 
        }
    }

    ofstream ofs(out, ios::binary); 
    if(!ofs){ cerr<<"Cannot open out file\n"; exit(1);} 
    boost::iostreams::filtering_streambuf<boost::iostreams::output> fb; 
    fb.push(boost::iostreams::zlib_compressor()); 
    fb.push(ofs); 
    boost::iostreams::copy(ss, fb);
}

void decompress(const string &in, const string &out){ 
    ifstream ifs(in, ios::binary); 
    if(!ifs){ cerr<<"Cannot open in file\n"; exit(1);} 
    stringstream ss; 
    boost::iostreams::filtering_streambuf<boost::iostreams::input> fb; 
    fb.push(boost::iostreams::zlib_decompressor()); 
    fb.push(ifs); 
    boost::iostreams::copy(fb, ss);
    
    int32_t w32=0, h32=0; 
    CompressionConfig cfg; 
    ss.read(reinterpret_cast<char*>(&w32), sizeof(int32_t)); 
    ss.read(reinterpret_cast<char*>(&h32), sizeof(int32_t)); 
    ss.read(reinterpret_cast<char*>(&cfg.spatial_block), sizeof(int32_t)); 
    ss.read(reinterpret_cast<char*>(&cfg.spatial_stride), sizeof(int32_t)); 
    ss.read(reinterpret_cast<char*>(&cfg.spectral_block), sizeof(int32_t)); 
    ss.read(reinterpret_cast<char*>(&cfg.spectral_coeffs), sizeof(int32_t)); 
    ss.read(reinterpret_cast<char*>(&cfg.quant_step), sizeof(float)); 
    cfg.validate(); 
    
    int w = static_cast<int>(w32), h = static_cast<int>(h32);
    uint32_t s_count = 0; 
    ss.read(reinterpret_cast<char*>(&s_count), sizeof(uint32_t)); 
    vector<MatrixXf> s_atoms(s_count, MatrixXf(cfg.spatial_block, cfg.spatial_block)); 
    for(uint32_t i=0; i<s_count; ++i) ss.read(reinterpret_cast<char*>(s_atoms[i].data()), static_cast<std::streamsize>(s_atoms[i].size()*sizeof(float)));
    
    uint32_t sp_count = 0; 
    ss.read(reinterpret_cast<char*>(&sp_count), sizeof(uint32_t)); 
    vector<MatrixXf> sp_atoms(sp_count, MatrixXf(cfg.spectral_coeffs, cfg.spectral_coeffs)); 
    for(uint32_t i=0; i<sp_count; ++i) ss.read(reinterpret_cast<char*>(sp_atoms[i].data()), static_cast<std::streamsize>(sp_atoms[i].size()*sizeof(float)));
    
    Image img; img.width=w; img.height=h; img.data.setZero(h,w); 
    MatrixXf weight = MatrixXf::Zero(h,w);
    
    for(int i=0; i+cfg.spatial_block <= h; i += cfg.spatial_stride) {
        for(int j=0; j+cfg.spatial_block <= w; j += cfg.spatial_stride){ 
            FieldEntry e; 
            ss.read(reinterpret_cast<char*>(&e), sizeof(FieldEntry)); 
            if(e.id < s_atoms.size()){ 
                MatrixXf atom = apply_galois_action(s_atoms[e.id], e.morphism); 
                img.data.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += (atom.array() * (static_cast<float>(e.chroma)*2.0f)) + static_cast<float>(e.luma); 
            } else {
                img.data.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += static_cast<float>(e.luma); 
            }
            weight.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += 1.0f; 
        }
    }
    img.data = img.data.array() / weight.array().max(1.0f);
    
    int sbk = cfg.spectral_block; 
    int sc = cfg.spectral_coeffs; 
    MatrixXf T(sbk,sbk); 
    for(int r=0;r<sbk;++r) for(int c=0;c<sbk;++c) T(r,c) = (r==0)? static_cast<float>(1.0/sqrt((double)sbk)) : static_cast<float>(sqrt(2.0/sbk)*cos((2*c+1)*r*M_PI/(2.0*sbk)));
    
    for(int i=0; i+sbk <= h; i += sbk) {
        for(int j=0; j+sbk <= w; j += sbk){ 
            FieldEntry e; 
            ss.read(reinterpret_cast<char*>(&e), sizeof(FieldEntry)); 
            MatrixXf coeff = MatrixXf::Zero(sbk,sbk); 
            if(e.id < sp_atoms.size()) coeff.block(0,0,sc,sc) = apply_spectral_action(sp_atoms[e.id], e.morphism) * (static_cast<float>(e.chroma) * cfg.quant_step); 
            img.data.block(i,j,sbk,sbk) += T.transpose() * coeff * T; 
        }
    }
    savePGM(out, img);
}

int main(int argc, char** argv){ 
    if(argc < 4){ 
        cout << "Usage: " << argv[0] << " <mode: c/d> <input> <output> [spatial_block] [stride] [spectral_block] [coeffs] [quant] [spatial_entries] [spectral_entries] [min_variance]\n"; 
        return 1;
    } 

    CompressionConfig cfg;
    string mode = argv[1];
    string in_file = argv[2];
    string out_file = argv[3];

    try {
        if(argc >= 5) cfg.spatial_block = stoi(argv[4]); 
        if(argc >= 6) cfg.spatial_stride = stoi(argv[5]); 
        if(argc >= 7) cfg.spectral_block = stoi(argv[6]); 
        if(argc >= 8) cfg.spectral_coeffs = stoi(argv[7]); 
        if(argc >= 9) cfg.quant_step = stof(argv[8]); 
        if(argc >= 10) cfg.spatial_entries = stoi(argv[9]);
        if(argc >= 11) cfg.spectral_entries = stoi(argv[10]);
        if(argc >= 12) cfg.dict_min_variance = stof(argv[11]);
    } catch (const exception& e) {
        cerr << "Parameter parsing error: " << e.what() << "\n";
        return 1;
    }

    cfg.validate(); 

    if(mode == "c") {
        compress(in_file, out_file, cfg);
    } else if(mode == "d") {
        decompress(in_file, out_file);
    } else {
        cerr << "Unknown mode: " << mode << " (use 'c' for compress, 'd' for decompress)\n";
        return 1;
    }
    
    return 0;
}
