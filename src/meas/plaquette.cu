

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>

#include <meas/plaquette.h>
#include <device_load_save.h>
#include <cuda_common.h>
#include <constants.h>
#include <index.h>
#include <reduction.h>
#include <timer.h>
#include <texture_host.h>
#include <comm_mpi.h>


#include <cudaAtomic.h>

#include <reduce_block_1d.h>

using namespace std;


namespace CULQCD{


//kernel to calculate the plaquette at each site of the lattice in EvenOdd order 
template <bool UseTex, ArrayType atype, class Real> 
__global__ void kernel_calc_plaquette_evenodd(complex *array, complex *plaquette ){
	uint idd = INDEX1D();
	
	
	complex plaq = complex::zero();
	if(idd < param_Volume()) {

		int oddbit = 0;
		int id = idd;
		if(idd >= param_HalfVolume()){
			oddbit = 1;
			id = idd - param_HalfVolume();
		}
		#ifdef MULTI_GPU
			int x[4];
			Index_4D_EO(x, id, oddbit);
			for(int i=0; i<4;i++) x[i] += param_border(i);
			int idxoddbit = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) >> 1 ;
			idxoddbit += oddbit  * param_HalfVolumeG();
			int mustride = DEVPARAMS::VolumeG;
			int offset = mustride * 4;
		#else
			int mustride = DEVPARAMS::Volume;
			int offset = mustride * 4;
			int idxoddbit = id + oddbit  * param_HalfVolume();
			//int idxoddbit = idd; //cuda reports error: misaligned address LOL

		#endif
		//------------------------------------------------------------------------
		// Calculate space-time plaquettes, stored in the real real of plaquette array
		//------------------------------------------------------------------------
		msun link, link1;
		//#pragma unroll
		for(int mu = 0; mu < 3; mu++){	
			link1 = GAUGE_LOAD<UseTex, atype,Real>( array, idxoddbit + mu * mustride, offset);
			int newidmu1 = Index_4D_Neig_EO(id, oddbit, mu, 1);
			//#pragma unroll
			for (int nu = (mu+1); nu < 4; nu++){
				link = GAUGE_LOAD<UseTex, atype,Real>( array,  newidmu1 + nu * mustride, offset);	      
				link *= GAUGE_LOAD_DAGGER<UseTex, atype,Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + mu * mustride, offset);			
				link *= GAUGE_LOAD_DAGGER<UseTex, atype,Real>( array, idxoddbit + nu * mustride, offset);
				if(nu == 3) plaq.imag() += (link1 * link).realtrace();
				else plaq.real() += (link1 * link).realtrace();
			}
		}
	}
    reduce_block_1d<complex>(plaquette, plaq);	  
}





template <class Real> 
Plaquette<Real>::Plaquette(gauge &array):array(array){
	plaq_value = complex::zero();
	size = 1;
	for(int i=0;i<4;i++){
		grid[i]=PARAMS::Grid[i];
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;
    reduced = false;
    SetFunctionPtr();
	sum = (complex*) dev_malloc(sizeof(complex));
}
template <class Real> 
void Plaquette<Real>::SetFunctionPtr(){
	tex = PARAMS::UseTex;
	kernel_pointer = NULL;
	if(array.EvenOdd()){
	    if(tex){
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_pointer = &kernel_calc_plaquette_evenodd<true, SOA, Real>;		
	        if(array.Type() == SOA12) kernel_pointer = &kernel_calc_plaquette_evenodd<true, SOA12, Real>;
	        if(array.Type() == SOA8) kernel_pointer = &kernel_calc_plaquette_evenodd<true, SOA8, Real>;
			#else
	        kernel_pointer = &kernel_calc_plaquette_evenodd<true, SOA, Real>;	
			#endif
	    }
	    else{
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_pointer = &kernel_calc_plaquette_evenodd<false, SOA, Real>;
	        if(array.Type() == SOA12) kernel_pointer = &kernel_calc_plaquette_evenodd<false, SOA12, Real>;
	        if(array.Type() == SOA8) kernel_pointer = &kernel_calc_plaquette_evenodd<false, SOA8, Real>;
			#else
	        kernel_pointer = &kernel_calc_plaquette_evenodd<false, SOA, Real>;	
			#endif
	    }
	}
	if(kernel_pointer == NULL) errorCULQCD("No kernel plaquette function exist for this gauge array...");
}


template <class Real> 
void Plaquette<Real>::apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
 	  CUDA_SAFE_CALL(cudaMemset(sum, 0, sizeof(complex)));
      kernel_pointer<<<tp.grid,tp.block, tp.block.x*sizeof(complex), stream>>>(array.GetPtr(), sum);
      reduced = false;
}
template <class Real> 
complex Plaquette<Real>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    plaqtime.start();
#endif
    //just ensure that the texture was not unbind somewhere...
    if(tex != PARAMS::UseTex){
    	SetFunctionPtr();
    } 
    GAUGE_TEXTURE(array.GetPtr(), true);
    apply(stream);
    CUDA_SAFE_CALL(cudaMemcpy(&plaq_value, sum, sizeof(complex), cudaMemcpyDeviceToHost));	
	plaq_value /= (Real)(3 * NCOLORS * size);
	#ifdef MULTI_GPU
	comm_Allreduce(&plaq_value);
	plaq_value /= numnodes();
	#endif
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    CUT_CHECK_ERROR("Kernel execution failed");
    plaqtime.stop();
    timesec = plaqtime.getElapsedTimeInSec();
#endif
	return plaq_value;
}
template <class Real> 
complex Plaquette<Real>::Run(){
	return Run(0);
}
template <class Real> 
complex Plaquette<Real>::Value() {
	return plaq_value;
}

template <class Real> 
double Plaquette<Real>::time(){
	return timesec;
}

template <class Real> 
void Plaquette<Real>::stat(){
	COUT << "Plaquette:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
}


template <class Real> 
void Plaquette<Real>::printValue(){ 
	Real resf = (plaq_value.real() + plaq_value.imag()) / 2.0;
	//.real() space  and .imag() time part
	printfCULQCD("Plaquette: < %.12e : %.12e > :: mean: %.12e\n", plaq_value.real(), plaq_value.imag(),resf);
}



template <class Real> 
long long Plaquette<Real>::flop() const { 
    //NEED TO RECOUNT!!!!!! 
	#ifdef MULTI_GPU
	return (array.getNumFlop(true) + NCOLORS * NCOLORS * NCOLORS * 120LL) * size * numnodes();
	#else
	return NCOLORS * NCOLORS * NCOLORS * 120LL * size * numnodes();
	#endif
}
template <class Real> 
long long Plaquette<Real>::bytes() const {
    //NEED TO RECOUNT!!!!!!  
	#ifdef MULTI_GPU
	return (22LL * array.getNumParams() + 4LL) * size * numnodes() * sizeof(Real);
	#else
	return (22LL * array.getNumParams() + 4LL) * size * sizeof(Real);
	#endif
}

template <class Real> 
double Plaquette<Real>::flops(){
	return ((double)flop() * 1.0e-9) / timesec;
}
template <class Real> 
double Plaquette<Real>::bandwidth(){
	return (double)bytes() / (timesec * (double)(1 << 30));
}

template <class Real> 
Plaquette<Real>::~Plaquette(){
	dev_free(sum);
}


template class Plaquette<float>;
template class Plaquette<double>;




}
