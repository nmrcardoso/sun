
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>

#ifdef MULTI_GPU
#include <mpi.h>
#endif

#include <timer.h>
#include <cuda_common.h>
#include <monte/ovr.h>
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


#include "staple.cuh"

using namespace std;


namespace CULQCD{


/**
	@brief CUDA Kernel to perform overrelaxation in a even/odd lattice order.
	@param array gauge field
	@param parity if 0 update even lattice sites, if 1 update odd lattice sites
	@param mu lattice direction to update links 
*/
//TESLA FERMI: THIS GIVES BETTER PERFORMANCE FOR SOA and SOA8....
template <bool UseTex, ArrayType atype, class Real, bool stapletSOA12ype, int actiontype> 
__global__ void 
kernel_overrelaxation_evenodd(complex *array, complex *staple_array, int oddbit, int mu){
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
	if( actiontype == 1 || actiontype == 2 ){
		staple = GAUGE_LOAD<false, SOA, Real>( staple_array, id, param_HalfVolume());
	}
	else {
		if( stapletSOA12ype )
			Staple_SOA12<UseTex, atype, Real>(array, mu, staple, x, id, oddbit, idxoddbit, mustride, muvolume, offset);
		else
			Staple<UseTex, atype, Real>(array, mu, staple, id, oddbit, idxoddbit, mustride, muvolume, offset);
	}
    idxoddbit += muvolume;
	msun U = GAUGE_LOAD<UseTex, atype, Real>( array, idxoddbit, offset);
	overrelaxationSUN<Real>( U, staple.dagger() );
	GAUGE_SAVE<atype, Real>( array, U, idxoddbit, offset);
}



template <class Real, int actiontype> 
OverRelaxation<Real, actiontype>::OverRelaxation(gauge &array):array(array){
	SetFunctionPtr();
	size = 1;
	for(int i=0;i<4;i++){
		grid[i]=PARAMS::Grid[i];
		size *= PARAMS::Grid[i];
	} 
	size = size >> 1;
	timesec = 0.0;
	if( actiontype == 1 || actiontype == 2 ) staple = GetStapleArray<Real>();
}

template <class Real, int actiontype> 
OverRelaxation<Real, actiontype>::~OverRelaxation(){ FreeStapleArray(); }


template <class Real, int actiontype> 
void OverRelaxation<Real, actiontype>::SetFunctionPtr(){
	kernel_pointer = NULL;
	tex = PARAMS::UseTex;
	if(array.EvenOdd()){
	    if(tex){
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_pointer = &kernel_overrelaxation_evenodd<true, SOA, Real, false, actiontype>;		
	        if(array.Type() == SOA12) kernel_pointer = &kernel_overrelaxation_evenodd<true, SOA12, Real, true, actiontype>;
	        if(array.Type() == SOA8) kernel_pointer = &kernel_overrelaxation_evenodd<true, SOA8, Real, false, actiontype>;
			#else
	        kernel_pointer = &kernel_overrelaxation_evenodd<true, SOA, Real, false, actiontype>;	
			#endif
	    }
	    else{
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_pointer = &kernel_overrelaxation_evenodd<false, SOA, Real, false, actiontype>;
	        if(array.Type() == SOA12) kernel_pointer = &kernel_overrelaxation_evenodd<false, SOA12, Real, true, actiontype>;
	        if(array.Type() == SOA8) kernel_pointer = &kernel_overrelaxation_evenodd<false, SOA8, Real, false, actiontype>;
			#else
	        kernel_pointer = &kernel_overrelaxation_evenodd<false, SOA, Real, false, actiontype>;	
			#endif
	    }
	}
	if(kernel_pointer == NULL) errorCULQCD("No kernel OverRelaxation function exist for this gauge array...");
}

template <class Real, int actiontype> 
void OverRelaxation<Real, actiontype>::apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if(actiontype == 1 || actiontype == 2) 
     	kernel_pointer<<<tp.grid,tp.block, 0, stream>>>(array.GetPtr(), staple->GetPtr(), parity, dir);
  	  else
      	kernel_pointer<<<tp.grid,tp.block, 0, stream>>>(array.GetPtr(), 0, parity, dir);
}
template <class Real, int actiontype> 
void OverRelaxation<Real, actiontype>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    mtime.start();
#endif
    //just ensure that the texture was not unbind somewhere...
    if(tex != PARAMS::UseTex){
    	SetFunctionPtr();
    } 
    GAUGE_TEXTURE(array.GetPtr(), true);
	for(parity=0; parity < 2; parity++)
	for(dir = 0; dir < 4; dir++){
		if(actiontype==1) CalculateStaple(array, parity, dir, 1);	
		else if(actiontype==2) CalculateStaple(array, parity, dir, 2);
		apply(stream);	
		//EXCHANGE DATA!!!!!
	    #ifdef MULTI_GPU
	    if(numnodes()>1){
			CUDA_SAFE_DEVICE_SYNC( );
			Exchange_gauge_border_links_gauge(array, dir, parity);
		}
	    #endif
	}
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    CUT_CHECK_ERROR("Kernel execution failed");
    mtime.stop();
    timesec = mtime.getElapsedTimeInSec();
#endif
}
template <class Real, int actiontype> 
void OverRelaxation<Real, actiontype>::Run(){
	Run(0);
}
template <class Real, int actiontype> 
double OverRelaxation<Real, actiontype>::time(){
	return timesec;
}

template <class Real, int actiontype> 
void OverRelaxation<Real, actiontype>::stat(){
	COUT << "OverRelaxation:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
}


template <class Real, int actiontype> 
long long OverRelaxation<Real, actiontype>::flop() const {
	//NEEEDDDDD TO RECOUNT THIS PART!!!!!!!!!!!!!!!!!!!!!!!!! 
	#if (NCOLORS == 3)
	long long stapleflop = 2268LL ;
	long long ThreadFlop = (7LL * array.getNumFlop(true) + array.getNumFlop(false) + stapleflop + 843LL);
	#else
	long long overflop = NCOLORS * NCOLORS * NCOLORS + (NCOLORS * ( NCOLORS - 1) / 2) * (17+112 * NCOLORS);
    long long ThreadFlop = (NCOLORS * NCOLORS * NCOLORS * 84LL + overflop);
	#endif
	#ifdef MULTI_GPU
	return ThreadFlop * size * numnodes();
	#else
	return ThreadFlop * size;
	#endif
}
template <class Real, int actiontype> 
long long OverRelaxation<Real, actiontype>::bytes() const { 
    #ifdef MULTI_GPU
    return 20LL * array.getNumParams() * sizeof(Real) * size * numnodes();
	#else
    return 20LL * array.getNumParams() * sizeof(Real)  * size;	
	#endif
}



template <class Real, int actiontype> 
double OverRelaxation<Real, actiontype>::flops(){
	return ((double)flop() * 8 * 1.0e-9) / timesec;
}
template <class Real, int actiontype> 
double OverRelaxation<Real, actiontype>::bandwidth(){
	return (double)bytes() * 8 / (timesec * (double)(1 << 30));
}



template class OverRelaxation<float, 0>;
template class OverRelaxation<float, 1>;
template class OverRelaxation<float, 2>;
template class OverRelaxation<double, 0>;
template class OverRelaxation<double, 1>;
template class OverRelaxation<double, 2>;





}
