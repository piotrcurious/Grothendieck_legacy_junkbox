// improved_morphism_engine.cpp

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
#include <map>
#include <tuple>
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
    // Second residual layer parameters
    int rs2=8, re2=512;        // block size and atom limit for residual2
    float rvt2=5, lr2=80;      // variance threshold and lambda for residual2
};

#pragma pack(push,1)
struct Entry {
    uint16_t id=0, m=0; uint8_t off=0, gn=0;
    bool operator<(const Entry& o) const { return tie(id, m, off, gn) < tie(o.id, o.m, o.off, o.gn); }
    bool operator==(const Entry& o) const { return tie(id, m, off, gn) == tie(o.id, o.m, o.off, o.gn); }
};
#pragma pack(pop)

struct RDO {
    float ssd=FLT_MAX, bits=0;
    vector<uint8_t> flags;
    vector<Entry> entries;
};

class EntryVLC {
    vector<Entry> dict;
public:
    void train(const vector<Entry>& src) {
        map<Entry, int> counts;
        for(const auto& e : src) counts[e]++;
        vector<pair<int, Entry>> sorted;
        sorted.reserve(counts.size());
        for(auto& p : counts) sorted.push_back({p.second, p.first});
        sort(sorted.begin(), sorted.end(), [](auto& a, auto& b){ return a.first > b.first; });
        dict.clear();
        for(int i=0; i < min((int)sorted.size(), 254); ++i) dict.push_back(sorted[i].second);
    }

    void write(stringstream& ss, const vector<Entry>& entries) {
        uint8_t ds = (uint8_t)dict.size();
        ss.write((char*)&ds, 1);
        if(ds > 0) ss.write((char*)dict.data(), ds * sizeof(Entry));
        for(const auto& e : entries) {
            int idx = -1;
            for(int i=0; i<dict.size(); ++i) if(dict[i] == e) { idx = i; break; }
            if(idx >= 0) {
                uint8_t code = (uint8_t)idx;
                ss.write((char*)&code, 1);
            } else {
                uint8_t code = 255;
                ss.write((char*)&code, 1);
                ss.write((char*)&e, sizeof(Entry));
            }
        }
    }

    void read(stringstream& ss, vector<Entry>& out, int count) {
        uint8_t ds; ss.read((char*)&ds, 1);
        dict.resize(ds);
        if(ds > 0) ss.read((char*)dict.data(), ds * sizeof(Entry));
        out.reserve(count);
        for(int i=0; i<count; ++i) {
            uint8_t code; ss.read((char*)&code, 1);
            if(code < 255) {
                if(code < dict.size()) out.push_back(dict[code]);
                else out.push_back({});
            } else {
                Entry e; ss.read((char*)&e, sizeof(Entry));
                out.push_back(e);
            }
        }
    }
};

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
            else d(u,v) = sum;
        }
        if(inv) {
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

    // Now accepts an optional 'other' dictionary pointer and may return an Entry
    // whose high bit in id (0x8000) marks "from other".
    Entry solve(const MatrixXf &t, float &sc, const Dict* other = nullptr) {
        Entry b; sc=0; if(atoms.empty() && (other==nullptr || other->atoms.empty())) return b;
        MatrixXf tc = resz(t, CS); float tn = tc.norm();
        if(tn < 1e-6) return b; tc /= tn; float bc = -1;
        // search own atoms
        for(uint16_t i=0; i<atoms.size(); ++i) for(uint16_t m=0; m<16; ++m) {
            float c = (Engine::geom(atoms[i], m).array()*tc.array()).sum();
            if(c>bc) { bc=c; b.id=i; b.m=m; }
        }
        // search other dictionary (if provided). mark with high bit if selected
        if(other) {
            for(uint16_t i=0; i<other->atoms.size(); ++i) for(uint16_t m=0; m<16; ++m) {
                float c = (Engine::geom(other->atoms[i], m).array()*tc.array()).sum();
                if(c>bc) { bc=c; b.id = (uint16_t)(i | 0x8000); b.m=m; }
            }
        }
        sc = bc*tn; return b;
    }
};

RDO comp_qt(const MatrixXf &src, int r, int c, int sz, Dict &d, MatrixXf &rec, float lb, bool split, int ms, const Dict* other = nullptr) {
    RDO res; int he=min(sz, (int)src.rows()-r), we=min(sz, (int)src.cols()-c);
    if(he<=0 || we<=0) return res;
    MatrixXf blk = src.block(r,c,he,we); float mu=blk.mean(), sc=0, qs=1.5;
    Entry e = d.solve(blk.array()-mu, sc, other);
    e.off = (uint8_t)(clamp(mu+128.f, 0.f, 255.f)/4)*4;
    e.gn = (uint8_t)clamp(abs(sc)/qs, 0.f, 255.f);
    MatrixXf at = MatrixXf::Zero(sz,sz);
    // determine source atom vector (own or from other)
    bool from_other = (e.id & 0x8000);
    uint16_t idx = e.id & 0x7fff;
    if((!from_other && e.id < d.atoms.size()) || (from_other && other && idx < other->atoms.size())) {
        MatrixXf baseAtom;
        if(from_other) baseAtom = other->atoms[idx];
        else baseAtom = d.atoms[e.id];
        at = resz(Engine::geom(baseAtom, e.m), sz);
        if(sc<0) at = -at;
        if(at.norm()>1e-6) at *= (e.gn*qs/at.norm());
    }
    MatrixXf lr = (at.array()+(e.off-128.f)).matrix().block(0,0,he,we);
    float cost = (blk-lr).squaredNorm() + lb*sizeof(Entry)*8;
    if(split && sz>ms) {
        int h=sz/2; RDO ch[4]; float ss=0, sb=4;
        for(int i=0; i<4; i++) {
            ch[i] = comp_qt(src, r+(i/2)*h, c+(i%2)*h, h, d, rec, lb, 1, ms, other);
            ss+=ch[i].ssd; sb+=ch[i].bits;
        }
        if(ss+lb*sb < cost) {
            res.ssd=ss; res.bits=sb;
            res.flags.push_back(QS);
            for(int i=0; i<4; i++) {
                res.flags.insert(res.flags.end(), ch[i].flags.begin(), ch[i].flags.end());
                res.entries.insert(res.entries.end(), ch[i].entries.begin(), ch[i].entries.end());
            }
            return res;
        }
    }
    rec.block(r,c,he,we)=lr; res.ssd=(blk-lr).squaredNorm(); res.bits=sizeof(Entry)*8;
    if(split) res.flags.push_back(QL);
    res.entries.push_back(e);
    return res;
}

// Global iterators for recursion during decode
int flg_idx = 0;
int ent_idx = 0;

// dec_qt updated to accept optional alternate atoms pointer
void dec_qt(const vector<uint8_t>& flags, const vector<Entry>& entries, MatrixXf &t, int r, int c, int sz, const vector<MatrixXf> &ats, const vector<MatrixXf>* alt_ats = nullptr, bool split = true) {
    if(split) {
        if(flg_idx >= flags.size()) return;
        uint8_t f = flags[flg_idx++];
        if(f==QS) {
            int h=sz/2; for(int i=0; i<4; i++) dec_qt(flags, entries, t, r+(i/2)*h, c+(i%2)*h, h, ats, alt_ats, 1);
            return;
        }
    }
    if(ent_idx >= entries.size()) return;
    Entry e = entries[ent_idx++];
    int he=min(sz, (int)t.rows()-r), we=min(sz, (int)t.cols()-c);
    if(he<=0 || we<=0) return;
    MatrixXf at = MatrixXf::Constant(he,we,e.off-128.f);
    bool from_other = (e.id & 0x8000);
    uint16_t idx = e.id & 0x7fff;
    if(!from_other) {
        if(e.id < ats.size() && e.gn>0) {
            MatrixXf a = resz(Engine::geom(ats[e.id], e.m), sz);
            if(a.norm()>1e-6) at += (a.array()*(e.gn*1.5f/a.norm())).matrix().block(0,0,he,we);
        }
    } else {
        if(alt_ats && idx < alt_ats->size() && e.gn>0) {
            MatrixXf a = resz(Engine::geom((*alt_ats)[idx], e.m), sz);
            if(a.norm()>1e-6) at += (a.array()*(e.gn*1.5f/a.norm())).matrix().block(0,0,he,we);
        }
    }
    t.block(r,c,he,we) += at;
}

void process(string in, string out, bool dec, Config cfg={}) {
    stringstream bs;
    if(!dec) {
        ifstream fin(in, ios::binary); string m; int w, h, mv; fin >> m >> w >> h >> mv; fin.ignore();
        uint32_t mag=0x47444843; bs.write((char*)&mag,4); bs.write((char*)&w,4); bs.write((char*)&h,4);
        bs.write((char*)&cfg.bs,4); bs.write((char*)&cfg.rs,4);
        // Note: file format now includes a second-residual dictionary and entries after first residual
        vector<MatrixXf> ch(3, MatrixXf::Zero(h,w));
        for(int i=0; i<h; i++) for(int j=0; j<w; j++) for(int k=0; k<3; k++) { uint8_t v; fin.read((char*)&v,1); ch[k](i,j)=v; }

        for(int k=0; k<3; k++) {
            Dict bd, rd, rd2; MatrixXf rec=MatrixXf::Zero(h,w);
            bd.train(ch[k], cfg.bs, cfg.be, cfg.bvt); bd.ser(bs, 0);

            // Base Layer
            RDO r_base;
            for(int i=0; i<h; i+=cfg.bs) for(int j=0; j<w; j+=cfg.bs) {
                auto r = comp_qt(ch[k], i, j, cfg.bs, bd, rec, cfg.lb, 1, cfg.bs/2, nullptr);
                r_base.flags.insert(r_base.flags.end(), r.flags.begin(), r.flags.end());
                r_base.entries.insert(r_base.entries.end(), r.entries.begin(), r.entries.end());
            }
            uint32_t flen = r_base.flags.size(); bs.write((char*)&flen, 4);
            bs.write((char*)r_base.flags.data(), flen);
            EntryVLC vlc_b; vlc_b.train(r_base.entries); vlc_b.write(bs, r_base.entries);

            // Residual Layer (first residual)
            MatrixXf resi=ch[k]-rec, rr=MatrixXf::Zero(h,w);
            rd.train(resi, cfg.rs, cfg.re, cfg.rvt); rd.ser(bs, 0);

            RDO r_resid;
            // allow residual solver to use base dictionary atoms as alternate candidates
            for(int i=0; i<h; i+=cfg.rs) for(int j=0; j<w; j+=cfg.rs) {
                 auto r = comp_qt(resi, i, j, cfg.rs, rd, rr, cfg.lr, 0, 0, &bd);
                 if(!r.entries.empty()) r_resid.entries.push_back(r.entries[0]);
            }
            EntryVLC vlc_r; vlc_r.train(r_resid.entries); vlc_r.write(bs, r_resid.entries);

            // SECOND Residual Layer (residual of residual)
            MatrixXf res2 = resi - rr; // what's left after first residual reconstruction
            MatrixXf rr2 = MatrixXf::Zero(h,w);
            rd2.train(res2, cfg.rs2, cfg.re2, cfg.rvt2); rd2.ser(bs, 0);

            RDO r_resid2;
            // allow second residual solver to use first residual dictionary atoms as alternate candidates
            for(int i=0; i<h; i+=cfg.rs2) for(int j=0; j<w; j+=cfg.rs2) {
                 auto r = comp_qt(res2, i, j, cfg.rs2, rd2, rr2, cfg.lr2, 0, 0, &rd);
                 if(!r.entries.empty()) r_resid2.entries.push_back(r.entries[0]);
            }
            EntryVLC vlc_r2; vlc_r2.train(r_resid2.entries); vlc_r2.write(bs, r_resid2.entries);

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
            Dict bd, rd, rd2; bd.ser(bs,1);

            // Base Layer
            uint32_t flen; bs.read((char*)&flen, 4);
            vector<uint8_t> flags(flen); bs.read((char*)flags.data(), flen);

            vector<Entry> entries_b;
            int ent_count = 0; for(auto f:flags) if(f==QL) ent_count++;

            EntryVLC vlc_b; vlc_b.read(bs, entries_b, ent_count);

            flg_idx = 0; ent_idx = 0;
            for(int i=0; i<h; i+=bsz) for(int j=0; j<w; j+=bsz)
                dec_qt(flags, entries_b, ch[k], i, j, bsz, bd.atoms, nullptr, 1);

            // Residual Layer (first residual)
            rd.ser(bs,1);
            vector<Entry> entries_r;
            int r_rows = (h+rsz-1)/rsz, r_cols = (w+rsz-1)/rsz;
            EntryVLC vlc_r; vlc_r.read(bs, entries_r, r_rows*r_cols);

            ent_idx = 0;
            vector<uint8_t> dummy_flags;
            // For residual decode, pass bd.atoms as alternate atoms so entries that used base atoms can be decoded
            for(int i=0; i<h; i+=rsz) for(int j=0; j<w; j+=rsz)
                dec_qt(dummy_flags, entries_r, ch[k], i, j, rsz, rd.atoms, &bd.atoms, 0);

            // SECOND Residual Layer (decode)
            rd2.ser(bs,1);
            vector<Entry> entries_r2;
            int rsz2 = cfg.rs2; // note: cfg in this scope is default-constructed but decoding uses the previously-read rsz value for first residual; rs2 is independent
            int r2_rows = (h+rsz2-1)/rsz2, r2_cols = (w+rsz2-1)/rsz2;
            EntryVLC vlc_r2; vlc_r2.read(bs, entries_r2, r2_rows*r2_cols);

            ent_idx = 0;
            // For second residual decode, pass rd.atoms as alternate atoms so entries that used first-residual atoms can be decoded
            for(int i=0; i<h; i+=rsz2) for(int j=0; j<w; j+=rsz2)
                dec_qt(dummy_flags, entries_r2, ch[k], i, j, rsz2, rd2.atoms, &rd.atoms, 0);

        }
        for(int i=0; i<h; i++) for(int j=0; j<w; j++) for(int k=0; k<3; k++) {
            uint8_t v=clamp(ch[k](i,j), 0.f, 255.f); fo.write((char*)&v,1);
        }
    }
}


void help(const char* p) {
    cout << "GDHC v11.1 - VLC Compact Spectral Hybrid\n"
         << "Usage: " << p << " <c/d> <in> <out> [opts]\n\n"
         << "Opts:\n"
         << " --bs <int>  Base block (16)\n"
         << " --rs <int>  Resid block (8)\n"
         << " --rs2 <int>  Resid block 2 (8)\n"
         << " --be <int>  Base entries (256)\n"
         << " --re <int>  Resid entries (512)\n"
         << " --re2 <int>  Resid entries 2 (512)\n"
         << " --lb <float> Base lambda (250)\n"
         << " --lr <float> Resid lambda (120)\n"
         << " --lr2 <float> Resid lambda 2(120)\n"
         << " --bv <float> Base var (40)\n"
         << " --rv <float> Resid var (10)\n"
         << " --rv2 <float> Resid var 2 (10)\n";

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
        else if (a == "--rs2") c.rs2 = stoi(argv[++i]);
        else if (a == "--be") c.be = stoi(argv[++i]);
        else if (a == "--re") c.re = stoi(argv[++i]);
        else if (a == "--re2") c.re2 = stoi(argv[++i]);
        else if (a == "--lb") c.lb = stof(argv[++i]);
        else if (a == "--lr") c.lr = stof(argv[++i]);
        else if (a == "--lr2") c.lr2 = stof(argv[++i]);
        else if (a == "--bv") c.bvt = stof(argv[++i]);
        else if (a == "--rv") c.rvt = stof(argv[++i]);
        else if (a == "--rv2") c.rvt2 = stof(argv[++i]);

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
