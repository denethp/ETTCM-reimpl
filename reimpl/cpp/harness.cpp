// Per-event dump from the OFFICIAL ETTCM core (identity rectification -> use with synthetic .es files).
// Output columns: t_us  global_divergence  rho  nu_z   (paper sign convention)
// Build from inside the official "ettcm" folder after its normal cmake build, see reimpl/README.md.
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>
#include "ETTCM/global_decay.hpp"
#include "ETTCM/global_motion_inverse_depth.hpp"
#include "ETTCM/motion_model.hpp"
using namespace ETTCM;
struct Ident {
  uint16_t w, h; Matrix<float,3,3> K; std::vector<CvVector<float,2>> m;
  Ident(uint16_t w_, uint16_t h_, float f, float cx, float cy): w(w_), h(h_), m(w_*h_) {
    K.setIdentity(); K(0,0)=f; K(1,1)=f; K(0,2)=cx; K(1,2)=cy;
    for (int y=0;y<h;++y) for(int x=0;x<w;++x){ m[y*w+x][0]=x; m[y*w+x][1]=y; } }
  uint16_t width() const {return w;} uint16_t height() const {return h;}
  const Matrix<float,3,3>& intrinsics() const {return K;}
  const CvVector<float,2>& map(uint16_t x, uint16_t y) const {return m[y*w+x];}
};
struct REv { uint64_t t; uint16_t x_original, y_original; float x, y; bool is_increase; };
template <class M> int run(const char* in, const char* out, int s, float f, float cx, float cy, int W, int H) {
  Ident rect(W,H,f,cx,cy);
  Decay dec; dec.reset(10000);
  auto gd = make_global_decay<REv>(10000, [](REv e,float d,float n,float td,float r)->Decay{return {e.t,d,n,td,r};}, [&](Decay d){dec=d;});
  std::ofstream fo(out); float num=0, den=0;
  auto gm = make_global_motion_inverse_depth<REv, M>(s, 0.01f, 1000.f, 1.f, dec, rect,
    [](REv e, Vector<float,M::NVars> mv, Vector<float,M::NDims> v, float rho) {
       return std::make_tuple(e.t, mv, v, rho); },
    [&](auto tup) { float rho = std::get<3>(tup); auto mv = std::get<1>(tup);
       if (rho>0) { num = dec.decay*num - mv(M::NV-1)*rho; den = dec.decay*den + 1; }
       fo << std::get<0>(tup) << " " << (den>0?num/den:0) << " " << rho << " " << -mv(M::NV-1) << "\n"; });
  std::ifstream fin(in, std::ios::binary); std::vector<char> buf((std::istreambuf_iterator<char>(fin)), {});
  uint64_t t=0; size_t i=20;
  while (i < buf.size()) { uint8_t b=buf[i++]; if (b==0xFF){t+=127;continue;} if(b==0xFE) continue;
    t += b>>1; bool p=b&1; uint16_t x=(uint8_t)buf[i]|((uint8_t)buf[i+1]<<8); uint16_t y=(uint8_t)buf[i+2]|((uint8_t)buf[i+3]<<8); i+=4;
    REv e{t,x,y,(float)x,(float)y,p}; gd(e); gm(e); }
  return 0;
}
int main(int argc, char** argv) {
  if (argc < 10) { std::cerr << "usage: harness <scaling|translation|six_dof> in.es out.txt s f cx cy width height\n"; return 1; }
  std::string model=argv[1]; int s=atoi(argv[4]); float f=atof(argv[5]),cx=atof(argv[6]),cy=atof(argv[7]); int W=atoi(argv[8]),H=atoi(argv[9]);
  if (model=="scaling") return run<Scaling>(argv[2],argv[3],s,f,cx,cy,W,H);
  if (model=="translation") return run<Translation3D>(argv[2],argv[3],s,f,cx,cy,W,H);
  return run<ParameterizedSixDOF>(argv[2],argv[3],s,f,cx,cy,W,H);
}
