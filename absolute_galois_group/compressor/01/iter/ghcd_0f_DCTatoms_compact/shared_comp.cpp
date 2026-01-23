#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <cstdint>
#include <cfloat>
#include <random>
#include <Eigen/Dense>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

using namespace Eigen;
using namespace std;

const int CS = 8; 
enum QTType : uint8_t { QL = 0, QS = 1 };

struct Config {
    int bs=16, rs=8, be=256, re=512;
    float bvt=40, rvt=10, lb=250, lr=120;
};

#pragma pack(push,1)
struct Entry { uint16_t id=0, m=0; uint8_t off=0, gn=0; };
#pragma pack(pop)

struct RDO { float ssd=FLT_MAX, bits=0; string s; };

class Engine {
public:
    static MatrixXf geom(const MatrixXf& b, uint16_t m) {
        MatrixXf r;
        auto op = m & 7;
        if(op==0) r=b; else if(op==1) r=b.reverse();
        else if(op==2) r=b.rowwise().reverse(); else if(op==3) r=b.colwise().reverse();
        else if(op==4) r=b.transpose(); else if(op==5) r=b.transpose().reverse();
        else if(op==6) r=b.transpose().rowwise().reverse(); else r=b.transpose().colwise().reverse();
        return (m & 8) ? (-r).eval() : r;
    }

    static MatrixXf dct(const MatrixXf& s, bool inv) {
        int N = s.rows(); MatrixXf d(N,N);
        for(int u=0; u<N; ++u) for(int v=0; v<N; ++v) {
            float sum=0;
            for(int x=0; x<N; ++x) for(int y=0; y<N; ++y) {
                float a = (M_PI/N);
                if(!inv) sum += s(x,y)*cos(a*(x+.5)*u)*cos(a*(y+.5)*v);
                else {
                    float au = u?sqrt(2./N):sqrt(1./N), av = v?sqrt(2./N):sqrt(1./N);
                    sum += au*av*s(u,v)*cos(a*(x+.5)*u)*cos(a*(y+.5)*v);
                }
            }
            if(!inv) d(u,v) = ((u?sqrt(2./N):sqrt(1./N))*(v?sqrt(2./N):sqrt(1./N)))*sum;
            else d(u,v) = sum; // Logic inverted in loop for inv
        }
        if(inv) { // Standard IDCT correction
            d.setZero();
            for(int x=0; x<N; ++x) for(int y=0; y<N; ++y) {
                float sum=0;
                for(int u=0; u<N; ++u) for(int v=0; v<N; ++v)
                    sum += (u?sqrt(2./N):sqrt(1./N))*(v?sqrt(2./N):sqrt(1./N))*s(u,v)*cos((M_PI/N)*(x+.5)*u)*cos((M_PI/N)*(y+.5)*v);
                d(x,y)=sum;
            }
        }
        return d;
    }
};

MatrixXf resz(const MatrixXf& s, int t) {
    if(s.rows()==t) return s;
    MatrixXf d=MatrixXf::Zero(t,t);
    for(int r=0; r<t; ++r) for(int c=0; c<t; ++c)
        d(r,c) = s(min((int)(r*s.rows()/t),(int)s.rows()-1), min((int)(c*s.cols()/t),(int)s.cols()-1));
    return d;
}

struct Dict {
    vector<MatrixXf> atoms;
    void train(const MatrixXf &d, int bs, int me, float mv) {
        atoms.clear(); vector<pair<MatrixXf,float>> p;
        for(int i=0; i<=d.rows()-bs; i+=bs) for(int j=0; j<=d.cols()-bs; j+=bs) {
            MatrixXf b = d.block(i,j,bs,bs), c = b.array()-b.mean();
            float v = c.squaredNorm(); if(v<mv) continue;
            MatrixXf cn = resz(c, CS); float mx=-FLT_MAX; uint16_t bm=0;
            for(uint16_t m=0; m<8; ++m) {
                MatrixXf cand = Engine::geom(cn,m);
                float e = cand(0,0)+cand(0,1)*.5+cand(1,0)*.5;
                if(e>mx) { mx=e; bm=m; }
            }
            cn = Engine::geom(cn,bm);
            if(cn.norm()>1e-5) p.push_back({cn/cn.norm(), v});
        }
        sort(p.begin(), p.end(), [](auto& a, auto& b){return a.second > b.second;});
        for(auto& it : p) {
            if(atoms.size() >= (size_t)me) break;
            bool red=0; for(auto& a : atoms) {
                for(uint16_t m=0; m<16; m++) if(abs((Engine::geom(a,m).array()*it.first.array()).sum()) > .9) red=1;
            }
            if(!red) atoms.push_back(it.first);
        }
    }

    void ser(stringstream &ss, bool d) {
        if(!d) {
            uint32_t n=atoms.size(); ss.write((char*)&n, 4);
            for(auto &a : atoms) {
                MatrixXf co = Engine::dct(a,0);
                for(int i=0; i<CS; ++i) for(int j=0; j<CS; ++j) {
                    int8_t v = clamp(co(i,j)*127./(1.+(i+j)*.5), -128., 127.);
                    ss.write((char*)&v, 1);
                }
            }
        } else {
            uint32_t n; ss.read((char*)&n, 4); atoms.assign(n, MatrixXf::Zero(CS,CS));
            for(uint32_t k=0; k<n; ++k) {
                MatrixXf co(CS,CS);
                for(int i=0; i<CS; ++i) for(int j=0; j<CS; ++j) {
                    int8_t v; ss.read((char*)&v, 1);
                    co(i,j) = v * (1.+(i+j)*.5) / 127.;
                }
                atoms[k] = Engine::dct(co, 1);
            }
        }
    }

    Entry solve(const MatrixXf &t, float &sc) {
        Entry b; sc=0; if(atoms.empty()) return b;
        MatrixXf tc = resz(t, CS); float tn = tc.norm();
        if(tn < 1e-6) return b; tc /= tn; float bc = -1;
        for(uint16_t i=0; i<atoms.size(); ++i) for(uint16_t m=0; m<16; ++m) {
            float c = (Engine::geom(atoms[i], m).array()*tc.array()).sum();
            if(c>bc) { bc=c; b.id=i; b.m=m; }
        }
        sc = bc*tn; return b;
    }
};

RDO comp_qt(const MatrixXf &src, int r, int c, int sz, Dict &d, MatrixXf &rec, float lb, bool split, int ms) {
    RDO res; int he=min(sz, (int)src.rows()-r), we=min(sz, (int)src.cols()-c);
    if(he<=0 || we<=0) return res;
    MatrixXf blk = src.block(r,c,he,we); float mu=blk.mean(), sc=0, qs=1.5;
    Entry e = d.solve(blk.array()-mu, sc);
    e.off = (uint8_t)(clamp(mu+128.f, 0.f, 255.f)/4)*4;
    e.gn = (uint8_t)clamp(abs(sc)/qs, 0.f, 255.f);
    MatrixXf at = MatrixXf::Zero(sz,sz);
    if(e.id<d.atoms.size() && e.gn>0) {
        at = resz(Engine::geom(d.atoms[e.id], e.m), sz);
        if(sc<0) at = -at;
        if(at.norm()>1e-6) at *= (e.gn*qs/at.norm());
    }
    MatrixXf lr = (at.array()+(e.off-128.f)).matrix().block(0,0,he,we);
    float cost = (blk-lr).squaredNorm() + lb*sizeof(Entry)*8;
    if(split && sz>ms) {
        int h=sz/2; RDO ch[4]; float ss=0, sb=4;
        for(int i=0; i<4; i++) {
            ch[i] = comp_qt(src, r+(i/2)*h, c+(i%2)*h, h, d, rec, lb, 1, ms);
            ss+=ch[i].ssd; sb+=ch[i].bits;
        }
        if(ss+lb*sb < cost) {
            res.ssd=ss; res.bits=sb; uint8_t f=QS; res.s.append((char*)&f,1);
            for(int i=0; i<4; i++) res.s.append(ch[i].s); return res;
        }
    }
    rec.block(r,c,he,we)=lr; res.ssd=(blk-lr).squaredNorm(); res.bits=sizeof(Entry)*8;
    if(split){uint8_t f=QL; res.s.append((char*)&f,1);}
    res.s.append((char*)&e, sizeof(Entry)); return res;
}

void dec_qt(stringstream &ss, MatrixXf &t, int r, int c, int sz, const vector<MatrixXf> &ats, bool split) {
    if(split) {
        uint8_t f=0; if(!ss.read((char*)&f,1)) return;
        if(f==QS) {
            int h=sz/2; for(int i=0; i<4; i++) dec_qt(ss, t, r+(i/2)*h, c+(i%2)*h, h, ats, 1);
            return;
        }
    }
    Entry e; if(!ss.read((char*)&e, sizeof(Entry))) return;
    int he=min(sz, (int)t.rows()-r), we=min(sz, (int)t.cols()-c);
    if(he<=0 || we<=0) return;
    MatrixXf at = MatrixXf::Constant(he,we,e.off-128.f);
    if(e.id<ats.size() && e.gn>0) {
        MatrixXf a = resz(Engine::geom(ats[e.id], e.m), sz);
        if(a.norm()>1e-6) at += (a.array()*(e.gn*1.5f/a.norm())).matrix().block(0,0,he,we);
    }
    t.block(r,c,he,we) += at;
}

void process(string in, string out, bool dec, Config cfg={}) {
    stringstream bs; 
    if(!dec) {
        // PPM Load (Simplified for tokens)
        ifstream fin(in, ios::binary); string m; int w, h, mv; fin >> m >> w >> h >> mv; fin.ignore();
        uint32_t mag=0x47444843; bs.write((char*)&mag,4); bs.write((char*)&w,4); bs.write((char*)&h,4);
        bs.write((char*)&cfg.bs,4); bs.write((char*)&cfg.rs,4);
        vector<MatrixXf> ch(3, MatrixXf::Zero(h,w));
        for(int i=0; i<h; i++) for(int j=0; j<w; j++) for(int k=0; k<3; k++) { uint8_t v; fin.read((char*)&v,1); ch[k](i,j)=v; }
        for(int k=0; k<3; k++) {
            Dict bd, rd; MatrixXf rec=MatrixXf::Zero(h,w);
            bd.train(ch[k], cfg.bs, cfg.be, cfg.bvt); bd.ser(bs, 0);
            for(int i=0; i<h; i+=cfg.bs) for(int j=0; j<w; j+=cfg.bs) {
                auto r=comp_qt(ch[k], i, j, cfg.bs, bd, rec, cfg.lb, 1, cfg.bs/2);
                uint32_t s=r.s.size(); bs.write((char*)&s,4); bs.write(r.s.data(),s);
            }
            MatrixXf resi=ch[k]-rec, rr=MatrixXf::Zero(h,w);
            rd.train(resi, cfg.rs, cfg.re, cfg.rvt); rd.ser(bs, 0);
            for(int i=0; i<h; i+=cfg.rs) for(int j=0; j<w; j+=cfg.rs)
                bs.write(comp_qt(resi, i, j, cfg.rs, rd, rr, cfg.lr, 0, 0).s.data(), sizeof(Entry));
        }
        ofstream fo(out, ios::binary); boost::iostreams::filtering_streambuf<boost::iostreams::output> z;
        z.push(boost::iostreams::zlib_compressor()); z.push(fo); boost::iostreams::copy(bs, z);
    } else {
        ifstream fi(in, ios::binary); boost::iostreams::filtering_streambuf<boost::iostreams::input> z;
        z.push(boost::iostreams::zlib_decompressor()); z.push(fi); boost::iostreams::copy(z, bs);
        uint32_t mag; bs.read((char*)&mag,4); if(mag!=0x47444843) return;
        int w,h,bsz,rsz; bs.read((char*)&w,4); bs.read((char*)&h,4); bs.read((char*)&bsz,4); bs.read((char*)&rsz,4);
        ofstream fo(out, ios::binary); fo << "P6\n" << w << " " << h << "\n255\n";
        vector<MatrixXf> ch(3, MatrixXf::Zero(h,w));
        for(int k=0; k<3; k++) {
            Dict bd, rd; bd.ser(bs,1);
            for(int i=0; i<h; i+=bsz) for(int j=0; j<w; j+=bsz) {
                uint32_t s; bs.read((char*)&s,4); string t(s,0); bs.read(&t[0],s);
                stringstream ss(t); dec_qt(ss, ch[k], i, j, bsz, bd.atoms, 1);
            }
            rd.ser(bs,1);
            for(int i=0; i<h; i+=rsz) for(int j=0; j<w; j+=rsz) dec_qt(bs, ch[k], i, j, rsz, rd.atoms, 0);
        }
        for(int i=0; i<h; i++) for(int j=0; j<w; j++) for(int k=0; k<3; k++) {
            uint8_t v=clamp(ch[k](i,j), 0.f, 255.f); fo.write((char*)&v,1);
        }
    }
}

void help(const char* p) {
    cout << "GDHC v10.6 - Compact Spectral Hybrid\n"
         << "Usage: " << p << " <c/d> <in> <out> [opts]\n\n"
         << "Opts:\n"
         << " --bs <int>  Base block (16)\n"
         << " --rs <int>  Resid block (8)\n"
         << " --be <int>  Base entries (256)\n"
         << " --re <int>  Resid entries (512)\n"
         << " --lb <float> Base lambda (250)\n"
         << " --lr <float> Resid lambda (120)\n"
         << " --bv <float> Base var (40)\n"
         << " --rv <float> Resid var (10)\n";
}

int main(int argc, char** argv) {
    if (argc < 4) { help(argv[0]); return 1; }

    string m = argv[1], in = argv[2], out = argv[3];
    Config c;

    for (int i = 4; i < argc; ++i) {
        string a = argv[i];
        if (i + 1 >= argc) break;
        if (a == "--bs") c.bs = stoi(argv[++i]);
        else if (a == "--rs") c.rs = stoi(argv[++i]);
        else if (a == "--be") c.be = stoi(argv[++i]);
        else if (a == "--re") c.re = stoi(argv[++i]);
        else if (a == "--lb") c.lb = stof(argv[++i]);
        else if (a == "--lr") c.lr = stof(argv[++i]);
        else if (a == "--bv") c.bvt = stof(argv[++i]);
        else if (a == "--rv") c.rvt = stof(argv[++i]);
    }

    try {
        bool dec = (m == "d");
        cout << (dec ? "Decompressing..." : "Compressing...") << endl;
        process(in, out, dec, c);
        cout << "Done." << endl;
    } catch (const exception& e) {
        cerr << "Err: " << e.what() << endl;
        return 1;
    }
    return 0;
}
