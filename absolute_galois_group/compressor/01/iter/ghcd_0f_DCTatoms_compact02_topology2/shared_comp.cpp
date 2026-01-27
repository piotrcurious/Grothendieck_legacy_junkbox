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
#include <zlib.h>

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
    // id: bits 0-13 atom index, bits 14-15 dictionary index
    // m: bits 0-3 transformation (8 geom * 2 sign), bit 15 HAS_NEXT
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
        uint32_t count = (uint32_t)entries.size();
        ss.write((char*)&count, 4);
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

    void read(stringstream& ss, vector<Entry>& out, int count_unused) {
        uint8_t ds; ss.read((char*)&ds, 1);
        dict.resize(ds);
        if(ds > 0) ss.read((char*)dict.data(), ds * sizeof(Entry));
        uint32_t count; ss.read((char*)&count, 4);
        out.reserve(count);
        for(uint32_t i=0; i<count; ++i) {
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
    if(s.rows()==t && s.cols()==t) return s;
    MatrixXf d(t,t);
    float sr = (float)s.rows()/t, sc = (float)s.cols()/t;
    for(int r=0; r<t; ++r) {
        float rf = (r + 0.5f) * sr - 0.5f;
        int r0 = (int)floor(rf), r1 = r0 + 1;
        float dr = rf - r0;
        r0 = std::max(0, std::min(r0, (int)s.rows()-1));
        r1 = std::max(0, std::min(r1, (int)s.rows()-1));
        for(int c=0; c<t; ++c) {
            float cf = (c + 0.5f) * sc - 0.5f;
            int c0 = (int)floor(cf), c1 = c0 + 1;
            float dc = cf - c0;
            c0 = std::max(0, std::min(c0, (int)s.cols()-1));
            c1 = std::max(0, std::min(c1, (int)s.cols()-1));
            float v00 = s(r0, c0), v01 = s(r0, c1), v10 = s(r1, c0), v11 = s(r1, c1);
            d(r,c) = (1-dr)*(1-dc)*v00 + (1-dr)*dc*v01 + dr*(1-dc)*v10 + dr*dc*v11;
        }
    }
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

    // Accepts optional fallback dictionaries and returns an Entry with dictionary index encoded in bits 14-15 of id.
    Entry solve(const MatrixXf &t, float &sc, const vector<const Dict*> &others = {}) {
        Entry b; sc=0;
        bool any_atoms = !atoms.empty();
        for(auto o : others) if(o && !o->atoms.empty()) any_atoms = true;
        if(!any_atoms) return b;

        MatrixXf tc = resz(t, CS); float tn = tc.norm();
        if(tn < 1e-6) return b; tc /= tn; float bc = -1;

        // search own atoms (dict index 0)
        for(uint16_t i=0; i<atoms.size(); ++i) for(uint16_t m=0; m<16; ++m) {
            float c = (Engine::geom(atoms[i], m).array()*tc.array()).sum();
            if(c>bc) { bc=c; b.id=i; b.m=m; }
        }
        // search other dictionaries
        for(size_t d_idx=0; d_idx < others.size(); ++d_idx) {
            const Dict* other = others[d_idx];
            if(!other) continue;
            uint16_t d_flag = (uint16_t)((d_idx + 1) << 14);
            for(uint16_t i=0; i<other->atoms.size(); ++i) for(uint16_t m=0; m<16; ++m) {
                float c = (Engine::geom(other->atoms[i], m).array()*tc.array()).sum();
                if(c>bc) { bc=c; b.id = (uint16_t)(i | d_flag); b.m=m; }
            }
        }
        sc = bc*tn; return b;
    }
};

RDO comp_qt(const MatrixXf &src, int r, int c, int sz, Dict &d, MatrixXf &rec, float lb, bool split, int ms, const vector<const Dict*> &others = {}) {
    RDO res; int he=min(sz, (int)src.rows()-r), we=min(sz, (int)src.cols()-c);
    if(he<=0 || we<=0) return res;
    MatrixXf blk = src.block(r,c,he,we); float mu=blk.mean(), sc=0, qs=1.5;

    // First atom search
    Entry e1 = d.solve(blk.array()-mu, sc, others);
    e1.off = (uint8_t)(clamp(mu+128.f, 0.f, 255.f)/4)*4;
    e1.gn = (uint8_t)clamp(abs(sc)/qs, 0.f, 255.f);

    auto get_atom_mat = [&](const Entry& e) {
        MatrixXf at = MatrixXf::Zero(sz,sz);
        int d_idx = e.id >> 14;
        uint16_t idx = e.id & 0x3fff;
        const Dict* target_dict = (d_idx == 0) ? &d : (d_idx-1 < (int)others.size() ? others[d_idx-1] : nullptr);
        if(target_dict && idx < target_dict->atoms.size()) {
            at = resz(Engine::geom(target_dict->atoms[idx], e.m & 0x7fff), sz);
            if(at.norm()>1e-6) at *= (e.gn*qs/at.norm());
        }
        return at;
    };

    MatrixXf at1 = get_atom_mat(e1);
    MatrixXf cur_rec = (at1.array()+(e1.off-128.f)).matrix();
    float best_ssd = (blk-cur_rec.block(0,0,he,we)).squaredNorm();
    float best_bits = sizeof(Entry)*8;
    float best_cost = best_ssd + lb*best_bits;
    vector<Entry> best_entries = {e1};

    // Try iterative combination (Morphism Engine improvement)
    MatrixXf residual = blk.array() - mu - at1.block(0,0,he,we).array();
    float sc2 = 0;
    Entry e2 = d.solve(residual, sc2, others);
    e2.gn = (uint8_t)clamp(abs(sc2)/qs, 0.f, 255.f);
    e2.off = 0; // Offset is only stored in the first entry

    if(e2.gn > 0) {
        MatrixXf at2 = get_atom_mat(e2);
        MatrixXf combined_rec = cur_rec + at2;
        float ssd2 = (blk - combined_rec.block(0,0,he,we)).squaredNorm();
        float bits2 = sizeof(Entry)*8*2;
        float cost2 = ssd2 + lb*bits2;
        if(cost2 < best_cost) {
            best_entries[0].m |= 0x8000; // HAS_NEXT bit
            best_entries.push_back(e2);
            best_ssd = ssd2;
            best_bits = bits2;
            best_cost = cost2;
            cur_rec = combined_rec;
        }
    }

    if(split && sz>ms) {
        int h=sz/2; RDO ch[4]; float ss=0, sb=8;
        for(int i=0; i<4; i++) {
            ch[i] = comp_qt(src, r+(i/2)*h, c+(i%2)*h, h, d, rec, lb, 1, ms, others);
            ss+=ch[i].ssd; sb+=ch[i].bits;
        }
        if(ss+lb*sb < best_cost) {
            res.ssd=ss; res.bits=sb;
            res.flags.push_back(QS);
            res.entries.clear();
            for(int i=0; i<4; i++) {
                res.flags.insert(res.flags.end(), ch[i].flags.begin(), ch[i].flags.end());
                res.entries.insert(res.entries.end(), ch[i].entries.begin(), ch[i].entries.end());
            }
            return res;
        }
    }
    rec.block(r,c,he,we)=cur_rec.block(0,0,he,we);
    res.ssd=best_ssd; res.bits=best_bits;
    if(split) res.flags.push_back(QL);
    res.entries = best_entries;
    return res;
}

// Global iterators for recursion during decode
int flg_idx = 0;
int ent_idx = 0;

// dec_qt updated to handle multiple entries per block and multiple fallback dictionaries
void dec_qt(const vector<uint8_t>& flags, const vector<Entry>& entries, MatrixXf &t, int r, int c, int sz, const vector<MatrixXf> &ats, const vector<const vector<MatrixXf>*> &fallbacks = {}, bool split = true) {
    if(split) {
        if(flg_idx >= (int)flags.size()) return;
        uint8_t f = flags[flg_idx++];
        if(f==QS) {
            int h=sz/2; for(int i=0; i<4; i++) dec_qt(flags, entries, t, r+(i/2)*h, c+(i%2)*h, h, ats, fallbacks, 1);
            return;
        }
    }

    int he=min(sz, (int)t.rows()-r), we=min(sz, (int)t.cols()-c);
    if(he<=0 || we<=0) return;

    MatrixXf block_sum = MatrixXf::Zero(he, we);
    bool first = true;
    while(ent_idx < (int)entries.size()) {
        Entry e = entries[ent_idx++];
        if(first) {
            block_sum.setConstant(e.off - 128.f);
            first = false;
        }

        int d_idx = e.id >> 14;
        uint16_t idx = e.id & 0x3fff;
        const vector<MatrixXf>* target_atoms = nullptr;
        if(d_idx == 0) target_atoms = &ats;
        else if(d_idx-1 < (int)fallbacks.size()) target_atoms = fallbacks[d_idx-1];

        if(target_atoms && idx < target_atoms->size() && e.gn > 0) {
            MatrixXf a = resz(Engine::geom((*target_atoms)[idx], e.m & 0x7fff), sz);
            if(a.norm()>1e-6) block_sum += (a.array()*(e.gn*1.5f/a.norm())).matrix().block(0,0,he,we);
        }

        if(!(e.m & 0x8000)) break; // HAS_NEXT bit not set
    }
    t.block(r,c,he,we) += block_sum;
}

void process(string in, string out, bool dec, Config cfg={}) {
    stringstream bs;
    if(!dec) {
        ifstream fin(in, ios::binary); string m; int w, h, mv; fin >> m >> w >> h >> mv; fin.ignore();
        uint32_t mag=0x47444843; bs.write((char*)&mag,4); bs.write((char*)&w,4); bs.write((char*)&h,4);
        bs.write((char*)&cfg.bs,4); bs.write((char*)&cfg.rs,4); bs.write((char*)&cfg.rs2,4);
        // Note: file format now includes a second-residual dictionary and entries after first residual
        vector<MatrixXf> ch(3, MatrixXf::Zero(h,w));
        for(int i=0; i<h; i++) for(int j=0; j<w; j++) for(int k=0; k<3; k++) { uint8_t v; fin.read((char*)&v,1); ch[k](i,j)=v; }

        for(int k=0; k<3; k++) {
            Dict bd, rd, rd2; MatrixXf rec=MatrixXf::Zero(h,w);
            bd.train(ch[k], cfg.bs, cfg.be, cfg.bvt); bd.ser(bs, 0);

            // Base Layer
            RDO r_base;
            for(int i=0; i<h; i+=cfg.bs) for(int j=0; j<w; j+=cfg.bs) {
                auto r = comp_qt(ch[k], i, j, cfg.bs, bd, rec, cfg.lb, 1, cfg.bs/2, {});
                r_base.flags.insert(r_base.flags.end(), r.flags.begin(), r.flags.end());
                r_base.entries.insert(r_base.entries.end(), r.entries.begin(), r.entries.end());
            }
            uint32_t flen = (uint32_t)r_base.flags.size(); bs.write((char*)&flen, 4);
            bs.write((char*)r_base.flags.data(), flen);
            EntryVLC vlc_b; vlc_b.train(r_base.entries); vlc_b.write(bs, r_base.entries);

            // Residual Layer (first residual)
            MatrixXf resi=ch[k]-rec, rr=MatrixXf::Zero(h,w);
            rd.train(resi, cfg.rs, cfg.re, cfg.rvt); rd.ser(bs, 0);

            RDO r_resid;
            vector<const Dict*> others1 = {&bd};
            for(int i=0; i<h; i+=cfg.rs) for(int j=0; j<w; j+=cfg.rs) {
                 auto r = comp_qt(resi, i, j, cfg.rs, rd, rr, cfg.lr, 1, cfg.rs/2, others1);
                 r_resid.flags.insert(r_resid.flags.end(), r.flags.begin(), r.flags.end());
                 r_resid.entries.insert(r_resid.entries.end(), r.entries.begin(), r.entries.end());
            }
            uint32_t rflen = (uint32_t)r_resid.flags.size(); bs.write((char*)&rflen, 4);
            bs.write((char*)r_resid.flags.data(), rflen);
            EntryVLC vlc_r; vlc_r.train(r_resid.entries); vlc_r.write(bs, r_resid.entries);

            // SECOND Residual Layer (residual of residual)
            MatrixXf res2 = resi - rr;
            MatrixXf rr2 = MatrixXf::Zero(h,w);
            rd2.train(res2, cfg.rs2, cfg.re2, cfg.rvt2); rd2.ser(bs, 0);

            RDO r_resid2;
            vector<const Dict*> others2 = {&bd, &rd};
            for(int i=0; i<h; i+=cfg.rs2) for(int j=0; j<w; j+=cfg.rs2) {
                 auto r = comp_qt(res2, i, j, cfg.rs2, rd2, rr2, cfg.lr2, 1, cfg.rs2/2, others2);
                 r_resid2.flags.insert(r_resid2.flags.end(), r.flags.begin(), r.flags.end());
                 r_resid2.entries.insert(r_resid2.entries.end(), r.entries.begin(), r.entries.end());
            }
            uint32_t r2flen = (uint32_t)r_resid2.flags.size(); bs.write((char*)&r2flen, 4);
            bs.write((char*)r_resid2.flags.data(), r2flen);
            EntryVLC vlc_r2; vlc_r2.train(r_resid2.entries); vlc_r2.write(bs, r_resid2.entries);
        }
        ofstream fo(out, ios::binary);
        string s = bs.str();
        uLongf dl = compressBound(s.size());
        vector<uint8_t> d(dl);
        if(compress(d.data(), &dl, (const Bytef*)s.data(), s.size()) == Z_OK)
            fo.write((char*)d.data(), dl);
    } else {
        ifstream fi(in, ios::binary | ios::ate);
        streamsize sz = fi.tellg();
        fi.seekg(0, ios::beg);
        vector<char> buf(sz);
        fi.read(buf.data(), sz);
        uLongf dl = 100 * 1024 * 1024; // 100MB max uncompressed
        vector<uint8_t> d(dl);
        if(uncompress(d.data(), &dl, (const Bytef*)buf.data(), sz) == Z_OK)
            bs.write((char*)d.data(), dl);
        uint32_t mag; bs.read((char*)&mag,4); if(mag!=0x47444843) return;
        int w,h,bsz,rsz,rsz2; bs.read((char*)&w,4); bs.read((char*)&h,4); bs.read((char*)&bsz,4); bs.read((char*)&rsz,4); bs.read((char*)&rsz2,4);
        ofstream fo(out, ios::binary); fo << "P6\n" << w << " " << h << "\n255\n";
        vector<MatrixXf> ch(3, MatrixXf::Zero(h,w));

        for(int k=0; k<3; k++) {
            Dict bd, rd, rd2; bd.ser(bs,1);

            // Base Layer
            uint32_t flen; bs.read((char*)&flen, 4);
            vector<uint8_t> flags(flen); bs.read((char*)flags.data(), flen);
            vector<Entry> entries_b;
            EntryVLC vlc_b; vlc_b.read(bs, entries_b, 0);

            flg_idx = 0; ent_idx = 0;
            for(int i=0; i<h; i+=bsz) for(int j=0; j<w; j+=bsz)
                dec_qt(flags, entries_b, ch[k], i, j, bsz, bd.atoms, {}, 1);

            // Residual Layer (first residual)
            rd.ser(bs,1);
            uint32_t rflen; bs.read((char*)&rflen, 4);
            vector<uint8_t> rflags(rflen); bs.read((char*)rflags.data(), rflen);
            vector<Entry> entries_r;
            EntryVLC vlc_r; vlc_r.read(bs, entries_r, 0);

            flg_idx = 0; ent_idx = 0;
            vector<const vector<MatrixXf>*> fallbacks1 = {&bd.atoms};
            for(int i=0; i<h; i+=rsz) for(int j=0; j<w; j+=rsz)
                dec_qt(rflags, entries_r, ch[k], i, j, rsz, rd.atoms, fallbacks1, 1);

            // SECOND Residual Layer (decode)
            rd2.ser(bs,1);
            uint32_t r2flen; bs.read((char*)&r2flen, 4);
            vector<uint8_t> r2flags(r2flen); bs.read((char*)r2flags.data(), r2flen);
            vector<Entry> entries_r2;
            EntryVLC vlc_r2; vlc_r2.read(bs, entries_r2, 0);

            flg_idx = 0; ent_idx = 0;
            vector<const vector<MatrixXf>*> fallbacks2 = {&bd.atoms, &rd.atoms};
            for(int i=0; i<h; i+=rsz2) for(int j=0; j<w; j+=rsz2)
                dec_qt(r2flags, entries_r2, ch[k], i, j, rsz2, rd2.atoms, fallbacks2, 1);
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
