

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>


#include <device_load_save.h>
#include <cuda_common.h>
#include <constants.h>
#include <index.h>
#include <reduction.h>
#include <timer.h>
#include <texture_host.h>
#include <comm_mpi.h>


#include <tune.h>

#include <projectlink.h>
#include <reunitlink.h>

using namespace std;


namespace CULQCD{



template<class Real>
struct APE2Arg{
  complex *arrayin;
  complex *arrayout;
  Real w;
  Real tol;
  int nhits;
};




template <bool UseTex, ArrayType atype, class Real>
__global__ void kernel_APE2_Space(APE2Arg<Real> arg, int mu){  
        
	int id = INDEX1D();
  if(id >= DEVPARAMS::Volume) return;
  
  int x[4];
  Index_4D_NM(id, x);
  int mustride = DEVPARAMS::Volume;
  int offset = DEVPARAMS::size;
  
  
	//for(int mu = 0; mu < 3; mu++){
    int muvolume = mu * mustride;
	msun link, link1;
	msun staple = msu3::zero();
	msun staple1 = msu3::zero();
	int newidmu1 = Index_4D_Neig_NM(id, mu, 1);
	for(int nu = 0; nu < 3; nu++)  if(mu != nu) {
		int nuvolume = nu * mustride;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin,  id + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin, Index_4D_Neig_NM(id, nu, 1) + muvolume, offset);
		link1 = link;	
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.arrayin, newidmu1 + nuvolume, offset);
		
		link1 *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin, Index_4D_Neig_NM(id, nu, 1, mu, 1) + muvolume, offset);	
		link1 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.arrayin, Index_4D_Neig_NM(id, mu, 2) + nuvolume, offset);
		staple += link;
		staple1 += link1;
		//DOWN	
		int newidnum1 = Index_4D_Neig_NM(id, nu, -1);
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.arrayin,  newidnum1 + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin, newidnum1  + muvolume, offset);
		link1 = link;
		link *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin, Index_4D_Neig_NM(id, mu, 1, nu,  -1) + nuvolume, offset);
		link1 *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin, Index_4D_Neig_NM(id, nu, -1, mu, 1)  + muvolume, offset);
		link1 *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin, Index_4D_Neig_NM(id, mu, 1, nu,  -1) + nuvolume, offset);
		staple += link;
		staple1 += link1;
	}
	msun U = GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin,  id + muvolume, offset);
	//U *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin,  Index_4D_Neig_NM(id, mu, 1) + muvolume, offset);
	link = U + staple * arg.w + staple1 * 0.5;
	//link /= ( 1.0 + 6.0 * arg.w );
if(0){
     /* Start with a unitarized version */
    U = link;
    reunit_link<Real>( &U );
//printf("%d ::: %lf\n",id,U.det().real());
}
#if (NCOLORS > 3)
	project_link<Real>( U, link, arg.nhits, arg.tol );
	GAUGE_SAVE<atype, Real>( arg.arrayout, U, id + muvolume, offset);
#else
if(1){	
  SU3project( link, arg.nhits );
	GAUGE_SAVE<atype, Real>( arg.arrayout, link, id + muvolume, offset);
}else{
	project_link<Real>( U, link, arg.nhits, arg.tol );
	GAUGE_SAVE<atype, Real>( arg.arrayout, U, id + muvolume, offset);
}
#endif
  //}
}
  
  





template <bool UseTex, ArrayType atypein, class Real> 
class ApplyAPE2: Tunable{
private:
   gauge arrayin;
   gauge arrayout;
   APE2Arg<Real> arg;
   int size;
   double timesec;
   int mu;
#ifdef TIMMINGS
    Timer mtime;
#endif
   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      kernel_APE2_Space<UseTex, atypein, Real><<<tp.grid,tp.block, 0, stream>>>(arg, mu);
	}
public:
   ApplyAPE2(gauge &arrayin, gauge & arrayout, Real w, int nhits, Real tol):arrayin(arrayin),arrayout(arrayout){
		size = 1;
		//Number of threads is equal to the number of space points!
		for(int i=0;i<4;i++){
		  size *= PARAMS::Grid[i];
		} 
		size = size;
		timesec = 0.0;
	  arg.arrayin = arrayin.GetPtr();
	  arg.arrayout = arrayout.GetPtr();
	  arg.nhits = nhits;
	  arg.tol = tol;
	  arg.w = w;
	  mu = 0;
	}
  ~ApplyAPE2(){};
  
  void SetDir(int muin){ mu = muin;};

	double time(){return timesec;}
	void stat(){ COUT << "ApplyAPE2:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
	long long flop() const { return 0;}
	long long bytes() const { return 0;}
	double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
	double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string typear = arrayin.ToStringArrayType()+arrayout.ToStringArrayType();
    return TuneKey(vol.str().c_str(), typeid(*this).name(), typear.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
	void preTune() { }
	void postTune() {}
	
	void Run(const cudaStream_t &stream){
	#ifdef TIMMINGS
	    mtime.start();
	#endif
	    apply(stream);    
	#ifdef TIMMINGS
		CUDA_SAFE_DEVICE_SYNC( );
	    mtime.stop();
	    timesec = mtime.getElapsedTimeInSec();
	#endif
	}
	void Run(){return Run(0);}
};




template<class Real>
void ApplyAPE2inSpace(gauge array, Real w, int steps, int nhits, Real tol){
  cout << "Apply APE2 Smearing in Space with w = " << w << " steps = " << steps << " nhit = " << nhits << " tol = " << tol << endl;
  if(array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  gauge arrayout(array.Type(), Device, 4*PARAMS::Volume, array.EvenOdd());
  arrayout.Copy(array);
  const ArrayType atypein = SOA;
  if(PARAMS::UseTex){
	  GAUGE_TEXTURE(array.GetPtr(), true);
    ApplyAPE2<true, atypein, Real> APE2(array, arrayout, w, nhits, tol);
    for(int st = 0; st < steps; st++){
      for(int mu = 0; mu < 3; mu++){
        APE2.SetDir(mu);
        APE2.Run();
      }
      array.Copy(arrayout);
    }
    APE2.stat();
  }
  else{
    ApplyAPE2<false, atypein, Real> APE2(array, arrayout, w, nhits, tol);
    for(int st = 0; st < steps; st++){
      for(int mu = 0; mu < 3; mu++){
        APE2.SetDir(mu);
        APE2.Run();
      }
      array.Copy(arrayout);
    }
    APE2.stat();
  }
  arrayout.Release();
}
template void ApplyAPE2inSpace<float>(gauges array, float w, int steps, int nhits, float tol);
template void ApplyAPE2inSpace<double>(gauged array, double w, int steps, int nhits, double tol);







template<class Real>
void ApplyAPE2inTime(gauge array, Real w, int steps, int nhits, Real tol){
  cout << "Apply APE2 Smearing in Time with w = " << w << " steps = " << steps << " nhit = " << nhits << " tol = " << tol << endl;
  if(array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  gauge arrayout(array.Type(), Device, 4*PARAMS::Volume, array.EvenOdd());
  arrayout.Copy(array);
  const ArrayType atypein = SOA;
  if(PARAMS::UseTex){
	  GAUGE_TEXTURE(array.GetPtr(), true);
    ApplyAPE2<true, atypein, Real> APE2(array, arrayout, w, nhits, tol);
    APE2.SetDir(3);
    for(int st = 0; st < steps; st++){
      APE2.Run();
      array.Copy(arrayout);
    }
    APE2.stat();
  }
  else{
    ApplyAPE2<false, atypein, Real> APE2(array, arrayout, w, nhits, tol);
    APE2.SetDir(3);
    for(int st = 0; st < steps; st++){
      APE2.Run();
      array.Copy(arrayout);
    }
    APE2.stat();
  }
  arrayout.Release();
}
template void ApplyAPE2inTime<float>(gauges array, float w, int steps, int nhits, float tol);
template void ApplyAPE2inTime<double>(gauged array, double w, int steps, int nhits, double tol);





}
