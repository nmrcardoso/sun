
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>

#ifdef MULTI_GPU
#include <mpi.h>
#endif

#include <timer.h>
#include <cuda_common.h>
#include <monte/monte.h>
#include <device_load_save.h>
#include <constants.h>
#include <matrixsun.h>
#include <gaugearray.h>
#include <index.h>
#include <device_PHB_OVR.h>
#include <reunitlink.h>
#include <staple.h>
#include <comm_mpi.h>
#include <exchange.h>
#include <texture_host.h>

using namespace std;


namespace CULQCD{



gauges *staple_array_sp;
gauged *staple_array_dp;
bool allocated_staple_array_sp = false;
bool allocated_staple_array_dp = false;

template <class Real>
gauge* GetStapleArray(){ }

template<> gauges* GetStapleArray<float>(){
	if( !allocated_staple_array_sp ) {
		staple_array_sp = new gauges(SOA, Device, param_HalfVolume(), true);	
		allocated_staple_array_sp = true;
		COUT << "Allocationg staple container in single precision..." << endl;
	}
	return staple_array_sp;
}
template<> gauged* GetStapleArray<double>(){
	if( !allocated_staple_array_dp ) {
		staple_array_dp = new gauged(SOA, Device, param_HalfVolume(), true);	
		allocated_staple_array_dp = true;
		COUT << "Allocationg staple container in double precision..." << endl;
	}
	return staple_array_dp;
}


void FreeStapleArray(){ 
	if(allocated_staple_array_sp) {
		staple_array_sp->Release();
		delete staple_array_sp;
		allocated_staple_array_sp = false;
	}
	if(allocated_staple_array_dp) {
		staple_array_dp->Release();
		delete staple_array_dp;
		allocated_staple_array_dp = false;
	}
}











template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real, int actiontype> 
__global__ void 
kernel_staple_evenodd(complex *array, complex* staple_array, int oddbit, int mu){
	int id = INDEX1D();
	if(id >= param_HalfVolume()) return;	
	#ifdef MULTI_GPU
		int x[4];
		Index_4D_EO(x, id, oddbit);
		for(int i=0; i<4;i++)x[i]+=param_border(i);
		int idxoddbit = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) >> 1 ;
		idxoddbit += oddbit  * param_HalfVolumeG();
		int mustride = DEVPARAMS::VolumeG;
		int muvolume = mu * mustride;
		int offset = mustride * 4;
	#else
		int x[4];
		Index_4D_EO(x, id, oddbit);
		int idxoddbit = id + oddbit  * param_HalfVolume();
		int mustride = DEVPARAMS::Volume;
		int muvolume = mu * mustride;
		int offset = DEVPARAMS::size;
	#endif
	msun staple = msun::zero();
	if( actiontype == 1)
		Staple_SI_SII<UseTex, atypein, Real, false>(array, mu, staple, id, oddbit, idxoddbit, mustride, muvolume, offset);
	else if( actiontype ==  2)
		Staple_SI_SII<UseTex, atypein, Real, true>(array, mu, staple, id, oddbit, idxoddbit, mustride, muvolume, offset);
	else Staple<UseTex, atypein, Real>(array, mu, staple, id, oddbit, idxoddbit, mustride, muvolume, offset);
	
	GAUGE_SAVE<atypeout, Real>( staple_array, staple, id, param_HalfVolume());
}
































template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real, int actiontype> 
class CalcStaple: Tunable{
private:
   gauge arrayin;
   int size;
   double timesec;
   int mu;
   int oddbit;
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
      kernel_staple_evenodd<UseTex, atypein, atypeout, Real, actiontype><<<tp.grid,tp.block, 0, stream>>>(arrayin.GetPtr(), GetStapleArray<Real>()->GetPtr(), oddbit, mu);
	}
public:
   CalcStaple(gauge &arrayin, int oddbit, int mu):arrayin(arrayin), oddbit(oddbit), mu(mu){
		size = 1;
		//Number of threads is equal to the number of space points!
		for(int i=0;i<4;i++){
		  size *= PARAMS::Grid[i];
		} 
		size = size /2 ;
		timesec = 0.0;
	}
  ~CalcStaple(){};

	double time(){return timesec;}
	void stat(){ COUT << "CalcStaple:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
    string typear = arrayin.ToStringArrayType() + GetStapleArray<Real>()->ToStringArrayType();
    return TuneKey(vol.str().c_str(), typeid(*this).name(), typear.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
	void preTune() {  }
	void postTune() {  }
	
	void Run(const cudaStream_t &stream){
	#ifdef TIMMINGS
	    mtime.start();
	#endif
	    apply(stream);
	#ifdef TIMMINGS
		CUDA_SAFE_DEVICE_SYNC( );
    CUT_CHECK_ERROR("Kernel execution failed");
	    mtime.stop();
	    timesec = mtime.getElapsedTimeInSec();
	#endif
	}
	void Run(){return Run(0);}
};




template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real, int actiontype> 
void CalculateStaple(gauge array, int oddbit, int mu){
	CalcStaple<UseTex, atypein, atypeout, Real, actiontype>  cstaple(array, oddbit, mu);
	cstaple.Run();
	//cstaple.stat();
}


template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real> 
void CalculateStaple(gauge array, int oddbit, int mu, int actiontype){
	if( actiontype == 1)
		CalculateStaple<UseTex, atypein, atypeout, Real, 1>(array, oddbit, mu);
	else if( actiontype == 2)
		CalculateStaple<UseTex, atypein, atypeout, Real, 2>(array, oddbit, mu);
	else
		CalculateStaple<UseTex, atypein, atypeout, Real, 0>(array, oddbit, mu);
}

template<bool UseTex, class Real>
void CalculateStaple(gauge array, int oddbit, int mu, int actiontype){
	#if (NCOLORS == 3)
    if(array.Type() == SOA) CalculateStaple<UseTex, SOA, SOA, Real>(array, oddbit, mu, actiontype);		
    if(array.Type() == SOA12) CalculateStaple<UseTex, SOA12, SOA, Real>(array, oddbit, mu, actiontype);	
    if(array.Type() == SOA8) CalculateStaple<UseTex, SOA8, SOA, Real>(array, oddbit, mu, actiontype);	
	#else
    CalculateStaple<UseTex, SOA, SOA, Real>(array, oddbit, mu, actiontype);	;	
	#endif
}



template <class Real> 
void CalculateStaple(gauge array, int oddbit, int mu, int actiontype){
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(array.GetPtr(), true);
    CalculateStaple<true, Real>(array, oddbit, mu, actiontype);
  }
  else CalculateStaple<false, Real>(array, oddbit, mu, actiontype);
}


template void CalculateStaple<float>(gauges array, int oddbit, int mu, int actiontype);
template void CalculateStaple<double>(gauged array, int oddbit, int mu, int actiontype);



}
