
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>



#include <timer.h>
#include <cuda_common.h>
#include <device_load_save.h>
#include <constants.h>
#include <matrixsun.h>
#include <gaugearray.h>
#include <index.h>
#include <device_PHB_OVR.h>
#include <reunitlink.h>

#include <comm_mpi.h>
#include <exchange.h>
#include <texture_host.h>

#include <sharedmemtypes.h>

#include <tune.h>
#include <launch_kernel.cuh>


#include <cudaAtomic.h>

#include <cub/cub.cuh>


#include <reduce_block_1d.h>

using namespace std;


namespace CULQCD{


//#define __WILSON_LOOP_USE_CUB__



template<class Real>
struct WLArgR{
  complex *gaugefield;
  complex *gaugefield_nosmear;
  complex *WLsp;
  complex *res;
  int radius;
};





template<bool UseTex, class Real, ArrayType atype>
__global__ void kernel_WilsonLineSP1(WLArgR<Real> arg){
    int id = INDEX1D();

    if(id >= DEVPARAMS::Volume) return;

    for(int mu = 0; mu < 3; mu++){
        msun link = msun::identity();
        for(int radius = 0; radius < arg.radius; radius++){
            link *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, mu, radius) + mu * DEVPARAMS::Volume, DEVPARAMS::size);
	        GAUGE_SAVE<atype, Real>( arg.WLsp, link, id + mu * DEVPARAMS::Volume + radius * DEVPARAMS::Volume * 3, DEVPARAMS::Volume * 3 * arg.radius );

            }
    }
}




#ifdef __WILSON_LOOP_USE_CUB__
template<int blockSize, bool UseTex, class Real, ArrayType atype>
__global__ void kernel_WilsonLoopSSP1(WLArgR<Real> arg){
  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
#else
template<bool UseTex, class Real, ArrayType atype>
__global__ void kernel_WilsonLoopSSP1(WLArgR<Real> arg){
#endif       

  int id = INDEX1D();
	

  for(int mu = 0; mu < 3; mu++)
  for(int nu = 0; nu < 3; nu++){
  	if(mu==nu) continue;
	int tdirvolume = nu * DEVPARAMS::Volume;
    
    for(int radius = 0; radius <= arg.radius; radius++){
        msun linkb = msun::identity();
        if(id < DEVPARAMS::Volume && radius > 0)
            linkb = GAUGE_LOAD<UseTex, atype, Real>( arg.WLsp, id + mu * DEVPARAMS::Volume + (radius-1) * DEVPARAMS::Volume * 3, DEVPARAMS::Volume * 3 * arg.radius);
        
	    msun t0 = msun::identity();
	    msun t1 = msun::identity();
	    for(int it = 0; it <= arg.radius; it++){

		    int idt = Index_4D_Neig_NM(id, nu, it);
		    msun linktop = msun::identity();
		    if(id < DEVPARAMS::Volume && radius > 0)
		        linktop = GAUGE_LOAD<UseTex, atype, Real>( arg.WLsp, idt + mu * DEVPARAMS::Volume + (radius-1) * DEVPARAMS::Volume * 3, DEVPARAMS::Volume * 3 * arg.radius); 
     
			  complex wl = complex::zero();
			  if(id < DEVPARAMS::Volume) wl = (linkb * t1 * linktop.dagger() * t0.dagger()).trace();
               
#ifdef __WILSON_LOOP_USE_CUB__
			  complex aggregate;
			  aggregate = BlockReduce(temp_storage).Reduce(wl, Summ<complex>());
			  if (threadIdx.x == 0) CudaAtomicAdd(arg.res + it + (arg.radius+1) * radius, aggregate);

#else
			  reduce_block_1d<complex>(arg.res + it + (arg.radius+1) * radius, wl);
#endif
			  __syncthreads();


		    if(id < DEVPARAMS::Volume && it < arg.radius){
			    t0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield_nosmear, idt + tdirvolume, DEVPARAMS::size);
			    t1 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield_nosmear, Index_4D_Neig_NM(idt, mu, radius) + tdirvolume, DEVPARAMS::size);
		    } 

     	 }
    }
  }
}


template <bool UseTex, class Real, ArrayType atype> 
class WilsonLineSP1: Tunable{
private:
   WLArgR<Real> arg;
   gauge array;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer WilsonLinetime;
#endif
	TuneParam tp;

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
        tp = tuneLaunch(*this, getTuning(), getVerbosity());
        kernel_WilsonLineSP1<UseTex, Real, atype><<<tp.grid,tp.block, 0, stream>>>(arg);
}
public:
   WilsonLineSP1(WLArgR<Real> arg, gauge array): arg(arg), array(array){
	size = 1;
	for(int i=0;i<4;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;  
}
   ~WilsonLineSP1(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    WilsonLinetime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    WilsonLinetime.stop();
    timesec = WilsonLinetime.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const {	return 0;}
   long long bytes() const{return 0;}
   double time(){	return timesec;}
   void stat(){	COUT << "WilsonLineSP:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string tmp = "None";
    return TuneKey(vol.str().c_str(), typeid(*this).name(), tmp.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() {  }
  void postTune() {    }

};






template <bool UseTex, class Real, ArrayType atype> 
class WilsonLoopSSSP1: Tunable{
private:
   WLArgR<Real> arg;
   gauge array;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer WilsonLoopSStime;

#endif
	TuneParam tp;

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.

   unsigned int minThreads() const { return size; }

   void apply(const cudaStream_t &stream){
        tp = tuneLaunch(*this, getTuning(), getVerbosity());
        CUDA_SAFE_CALL(cudaMemset(arg.res, 0, (arg.radius + 1) * (arg.radius+1) * sizeof(complex)));
#ifdef __WILSON_LOOP_USE_CUB__
        LAUNCH_KERNEL(kernel_WilsonLoopSSP1, tp, stream, arg, UseTex, Real, atype);
#else
		kernel_WilsonLoopSSP1<UseTex, Real, atype><<<tp.grid,tp.block, tp.block.x*sizeof(complex), stream>>>(arg);
#endif

}
public:
   WilsonLoopSSSP1(WLArgR<Real> arg, gauge array): arg(arg), array(array){
	size = 1;
	for(int i=0;i<4;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;  
}
   ~WilsonLoopSSSP1(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    WilsonLoopSStime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    WilsonLoopSStime.stop();
    timesec = WilsonLoopSStime.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const {	return 0;}
   long long bytes() const{return 0;}
   double time(){	return timesec;}
   void stat(){	COUT << "WilsonLoopSS:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string tmp = "None";
    return TuneKey(vol.str().c_str(), typeid(*this).name(), tmp.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() {  }
  void postTune() {    }

};








template<bool UseTex, class Real>
void WilsonLoopSS1(gauge array, gauge array_nosmear, complex *res, int radius){
  Timer mtime;
  mtime.start(); 
  WLArgR<Real> arg;
	arg.gaugefield = array.GetPtr();
  	arg.gaugefield_nosmear = array_nosmear.GetPtr();
	arg.res = (complex*) dev_malloc((radius+1)*(radius+1)*sizeof(complex));
	arg.radius = radius;

	gauge WLsp(SOA, Device, PARAMS::Volume * 3 * radius, false);
    arg.WLsp = WLsp.GetPtr();
  
  if(array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
  if(array_nosmear.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array_nosmear.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");



    WilsonLineSP1<UseTex, Real, SOA> wline(arg, array);
    WilsonLoopSSSP1<UseTex, Real, SOA> wlsp(arg, array);
    wline.Run();
    wline.stat();
    wlsp.Run();
    wlsp.stat();
    WLsp.Release();


  CUDA_SAFE_CALL(cudaMemcpy(res, arg.res, (arg.radius + 1) * (arg.radius+1) * sizeof(complex), cudaMemcpyDeviceToHost));
  dev_free(arg.res);
  for(int r = 0; r <= radius; r++)
  for(int it = 0; it <= radius; it++)
    res[it + r * (radius+1)] /= (Real)(PARAMS::Volume * 3 * NCOLORS);
  CUDA_SAFE_DEVICE_SYNC( );
  mtime.stop();
  COUT << "Time WilsonLoopSS:  " <<  mtime.getElapsedTimeInSec() << " s"  << endl;
}




template<class Real>
void WilsonLoopSS1(gauge array, gauge array_nosmear, complex *res, int radius){
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(array.GetPtr(), true);
    WilsonLoopSS1<true, Real>(array, array_nosmear, res, radius);
  }
  else WilsonLoopSS1<false, Real>(array, array_nosmear, res, radius);
}


template void WilsonLoopSS1<float>(gauges array, gauges array_nosmear, complexs *res, int radius);
template void WilsonLoopSS1<double>(gauged array, gauged array_nosmear, complexd *res, int radius);





































































template<class Real>
struct WLArgRnoInt{
  complex *gaugefield;
  complex *gaugefield_nosmear;
  complex *WLsp;
  complex *res;
  int radius[3];
  int Tmax;
  int mu;
  int nu;
};





template<bool UseTex, class Real, ArrayType atype>
__global__ void kernel_WilsonLoopRnoInt(WLArgRnoInt<Real> arg){
          

	int id = INDEX1D();
	
	
    //if(id==0) printf("%d::%d::%d::\n",arg.radius[0],arg.radius[1],arg.radius[2]);
    
	

	int tdirvolume = arg.nu * DEVPARAMS::Volume;

    msun linkb = msun::identity();
	int ids = id;
	if(id < DEVPARAMS::Volume){
		for(int dir = 0; dir < 3; dir++){
			int newdir = dir;//( arg.mu + dir ) % 3;
			for(int radius = 0; radius < arg.radius[newdir]; radius++){
				linkb *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + newdir * DEVPARAMS::Volume, DEVPARAMS::size);
				ids = Index_4D_Neig_NM(ids, newdir, 1);			
			}
		}
	}
        
    msun tl = msun::identity();
    msun tr = msun::identity();
    for(int it = 0; it <= arg.Tmax; it++){

    int idt = Index_4D_Neig_NM(id, arg.nu, it);
    int idl = idt;
    int idr = Index_4D_Neig_NM(ids, arg.nu, it);
    
    msun linktop = msun::identity();    
	if(id < DEVPARAMS::Volume){
		for(int dir = 0; dir < 3; dir++){
			int newdir = dir;//( arg.mu + dir ) % 3;
			for(int radius = 0; radius < arg.radius[newdir]; radius++){
				linktop *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idt + newdir * DEVPARAMS::Volume, DEVPARAMS::size);
				idt = Index_4D_Neig_NM(idt, newdir, 1);			
			}
		}
	}
    
	complex wl = complex::zero();
	if(id < DEVPARAMS::Volume){
	  wl = (linkb * tr * linktop.dagger() * tl.dagger()).trace();
	}
	reduce_block_1d<complex>(arg.res + it, wl);


	if(id < DEVPARAMS::Volume){
		tl *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield_nosmear, idl + tdirvolume, DEVPARAMS::size);
		tr *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield_nosmear, idr + tdirvolume, DEVPARAMS::size);
	} 

  }
   // }
  //}
}





template <bool UseTex, class Real, ArrayType atype> 
class WilsonLoopRnoInt: Tunable{
private:
   WLArgRnoInt<Real> arg;
   gauge array;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer WilsonLooptime;
#endif
	TuneParam tp;

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
        tp = tuneLaunch(*this, getTuning(), getVerbosity());
        CUDA_SAFE_CALL(cudaMemset(arg.res, 0, (arg.Tmax+1) * sizeof(complex)));
		kernel_WilsonLoopRnoInt<UseTex, Real, atype><<<tp.grid,tp.block, tp.block.x*sizeof(complex), stream>>>(arg);

}
public:
   WilsonLoopRnoInt(WLArgRnoInt<Real> arg, gauge array): arg(arg), array(array){
	size = 1;
	for(int i=0;i<4;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;  
}
   ~WilsonLoopRnoInt(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    WilsonLooptime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    WilsonLooptime.stop();
    timesec = WilsonLooptime.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const {	return 0;}
   long long bytes() const{return 0;}
   double time(){	return timesec;}
   void stat(){	COUT << "WilsonLoop:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string tmp = "None";
    return TuneKey(vol.str().c_str(), typeid(*this).name(), tmp.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() {  }
  void postTune() {    }

};





template<bool UseTex, class Real>
void WilsonLoopR(gauge array, gauge array_nosmear, complex *res, int radius[3], int Tmax, int mu, int nu){
  Timer mtime;
  mtime.start(); 
  WLArgRnoInt<Real> arg;
	arg.gaugefield = array.GetPtr();
	arg.gaugefield_nosmear = array_nosmear.GetPtr();
	arg.res = (complex*) dev_malloc((Tmax+1)*sizeof(complex));
	for(int i = 0; i < 3; i++) arg.radius[i] = radius[i];	
	arg.Tmax = Tmax;
	arg.mu = mu;
	arg.nu = nu;

   
  
  if(array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
  if(array_nosmear.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array_nosmear.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");


	WilsonLoopRnoInt<UseTex, Real, SOA> wl(arg, array);
	wl.Run();
	//wl.stat();


  CUDA_SAFE_CALL(cudaMemcpy(res, arg.res, (arg.Tmax+1) * sizeof(complex), cudaMemcpyDeviceToHost));
  dev_free(arg.res);
  for(int it = 0; it <= Tmax; it++)
    res[it] /= (Real)(PARAMS::Volume * NCOLORS);
  CUDA_SAFE_DEVICE_SYNC( );
  mtime.stop();
  COUT << "Time WilsonLoop:  " <<  mtime.getElapsedTimeInSec() << " s"  << endl;
}




template<class Real>
void WilsonLoopR(gauge array, gauge array_nosmear, complex *res, int radius[3], int Tmax, int mu, int nu){
  WilsonLoopR<false, Real>(array, array_nosmear, res, radius, Tmax, mu, nu);
}


template void WilsonLoopR<float>(gauges array, gauges array_nosmear, complexs *res, int radius[3], int Tmax, int mu, int nu);
template void WilsonLoopR<double>(gauged array, gauged array_nosmear, complexd *res, int radius[3], int Tmax, int mu, int nu);










}
