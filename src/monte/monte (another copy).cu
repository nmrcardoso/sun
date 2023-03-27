
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


template <bool UseTex, ArrayType atype, class Real>
__device__ inline void Staple_SOA12____(complex *array, int oddbit, int mu, msun &staple, int x[4], int id, int idxoddbit, int mustride, int muvolume, int offset){
	int newidmu1 = Index_4D_Neig_EO(id, oddbit, mu, 1);
	for(int nu = 0; nu < 4; nu++)  if(mu != nu) {
      	int dx[4] = {0, 0, 0, 0};
		msun link;	
		int nuvolume = nu * mustride;
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		dx[nu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + muvolume, offset);	
		dx[nu]--;
		dx[mu]++;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + nuvolume, offset);
		staple += link;

		dx[mu]--;
		dx[nu]--;
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG()  + muvolume, offset);
		dx[mu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + oddbit * param_HalfVolumeG() + nuvolume, offset);
		staple += link;
	}
}





template <bool UseTex, ArrayType atype, class Real>
__device__ inline void Staple_SOA12(complex *array, int oddbit, int mu, msun &staple, int x[4], int id, int idxoddbit, int mustride, int muvolume, int offset){
	int newidmu1 = Index_4D_Neig_EO(id, oddbit, mu, 1);
	//Normal staple!!!!!
	msun wsp = msun::zero();
	msun wst = msun::zero();
	msun tmp = msun::zero();
	// Staple 1x1
	for(int nu = 0; nu < 4; nu++)  if(mu != nu) {
      	int dx[4] = {0, 0, 0, 0};
		msun link;	
		int nuvolume = nu * mustride;
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		dx[nu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + muvolume, offset);	
		dx[nu]--;
		dx[mu]++;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + nuvolume, offset);
		tmp += link;

		dx[mu]--;
		dx[nu]--;
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG()  + muvolume, offset);
		dx[mu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + oddbit * param_HalfVolumeG() + nuvolume, offset);
		tmp += link;
		
		
		if( mu == 3 || nu == 3) wst += tmp;
		else wsp += tmp;
	}
	msun wsr = msun::zero();
	msun wstr = msun::zero();
	tmp = msun::zero();
	
	/*
	for(int nu = 0; nu < 4; nu++)  if(mu != nu) {
      	int dx[4] = {0, 0, 0, 0};
		msun link;	
		int nuvolume = nu * mustride;
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		dx[nu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + muvolume, offset);	
		dx[nu]--;
		dx[mu]++;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + nuvolume, offset);
		tmp += link;

		dx[mu]--;
		dx[nu]--;
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG()  + muvolume, offset);
		dx[mu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + oddbit * param_HalfVolumeG() + nuvolume, offset);
		tmp += link;		

		//if( mu == 3 || nu == 3) wstr += tmp;
		if( nu == 3) wstr += tmp;
		else wsr += tmp;
	}*/
	
	
	
	// Staple 2x1
	for(int nu = 0; nu < 4; nu++)  if(mu != nu) {
      	int dx[4] = {0, 0, 0, 0};
		msun link, link1;	
		int nuvolume = nu * mustride;
		//link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + muvolume, offset);
		dx[mu]++;
		int oddbit1 = (x[0] + 1 + x[1] + x[2] +x[3]) & 1;
		link = GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + oddbit1 * param_HalfVolumeG() + muvolume, offset);
		link1 = link;	
		dx[mu]++;
		oddbit1 = (x[0] + 2 + x[1] + x[2] +x[3]) & 1;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + oddbit1 * param_HalfVolumeG() + nuvolume, offset);
		dx[nu]++;
		dx[mu]--;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + oddbit1 * param_HalfVolumeG() + muvolume, offset);
		dx[mu]--;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + muvolume, offset);
		dx[mu]--;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + oddbit * param_HalfVolumeG() + nuvolume, offset);	
		tmp += link;
		
		dx[mu] = 0;
		dx[nu] = 0;
		link = link1;
		dx[mu] += 2;
		dx[nu]--;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + nuvolume, offset);
		dx[mu]--;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + oddbit * param_HalfVolumeG() + muvolume, offset);
		dx[mu]--;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(x,dx,DEVPARAMS::GridWGhost) + (1-oddbit) * param_HalfVolumeG() + nuvolume, offset);	
		tmp += link;		
		//if( mu == 3 || nu == 3) wstr += tmp;
		if( nu == 3) wstr += tmp;
		else wsr += tmp;
	}
	wsr = wsr.dagger();
	wstr = wstr.dagger();
	Real us2 = DEVPARAMS::us * DEVPARAMS::us;
	Real ut2 = DEVPARAMS::ut * DEVPARAMS::ut;
	Real us4 = us2 * us2;
	Real c0 = 5./(3. * us4);
	Real c1 = 4. / (3. * DEVPARAMS::aniso * DEVPARAMS::aniso * us2 * ut2);
	Real c2 = 1. / (12. * us2 * us4);
	Real c3 = 1. / (12. * DEVPARAMS::aniso * DEVPARAMS::aniso * us4 * ut2);
	staple = ( wsp * c0 + wst * c1 - wsr * c2 - wstr * c3) * DEVPARAMS::aniso;
	
}







































/**
	@brief CUDA Kernel to perform pseudo-heatbath in a even/odd lattice order.
	@param array gauge field
	@param state CUDA RNG array state
	@param parity if 0 update even lattice sites, if 1 update odd lattice sites
	@param mu lattice direction to update links 
*/
template <bool UseTex, ArrayType atype, class Real> 
__global__ void 
kernel_PHeatBath_evenodd(complex *array, cuRNGState *state, int oddbit, int mu){
	int id = INDEX1D();
	if(id >= DEVPARAMS::HalfVolume) return;

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
		int mustride = DEVPARAMS::Volume;
		int muvolume = mu * mustride;
		int offset = DEVPARAMS::size;
		int idxoddbit = id + oddbit  * param_HalfVolume();
	#endif

	msun staple = msu3::zero();
	int newidmu1 = Index_4D_Neig_EO(id, oddbit, mu, 1);
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		msun link;	
		int nuvolume = nu * mustride;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);	
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + nuvolume, offset);
		staple += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_EO(id, oddbit, nu, -1);
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, newidnum1  + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu,  -1) + nuvolume, offset);
		staple += link;
	}
	//Copy state to local memory for efficiency
    cuRNGState localState = state[ id ];
    idxoddbit += muvolume;
	msun U = GAUGE_LOAD<UseTex, atype, Real>( array, idxoddbit, offset);
	heatBathSUN<Real>( U, staple.dagger(), localState );
    state[ id ] = localState;
	GAUGE_SAVE<atype, Real>( array, U, idxoddbit, offset);
}


template <bool UseTex, ArrayType atype, class Real> 
__global__ void 
kernel_PHeatBath_evenodd_SOA12(complex *array, cuRNGState *state, int oddbit, int mu){
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
	msun staple = msu3::zero();
	Staple_SOA12<UseTex, atype, Real>(array, oddbit, mu, staple, x, id, idxoddbit, mustride, muvolume, offset);
	//Copy state to local memory for efficiency
    cuRNGState localState = state[ id ];
    idxoddbit += muvolume;
	msun U = GAUGE_LOAD<UseTex, atype, Real>( array, idxoddbit, offset);
	heatBathSUN<Real>( U, staple.dagger(), localState );
    state[ id ] = localState;
	GAUGE_SAVE<atype, Real>( array, U, idxoddbit, offset);
}







template <class Real> 
HeatBath<Real>::HeatBath(gauge &array, RNG &randstates):array(array), randstates(randstates){
	SetFunctionPtr();
	size = 1;
	for(int i=0;i<4;i++){
		grid[i]=PARAMS::Grid[i];
		size *= PARAMS::Grid[i];
	} 
	size = size >> 1;
	timesec = 0.0;
}

template <class Real> 
void HeatBath<Real>::SetFunctionPtr(){
	kernel_pointer = NULL;
	tex = PARAMS::UseTex;
	if(array.EvenOdd()){
	    if(tex){
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_pointer = &kernel_PHeatBath_evenodd<true, SOA, Real>;		
	        if(array.Type() == SOA12) kernel_pointer = &kernel_PHeatBath_evenodd_SOA12<true, SOA12, Real>;
	        if(array.Type() == SOA8) kernel_pointer = &kernel_PHeatBath_evenodd<true, SOA8, Real>;
			#else
	        kernel_pointer = &kernel_PHeatBath_evenodd<true, SOA, Real>;	
			#endif
	    }
	    else{
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_pointer = &kernel_PHeatBath_evenodd<false, SOA, Real>;
	        if(array.Type() == SOA12) kernel_pointer = &kernel_PHeatBath_evenodd_SOA12<false, SOA12, Real>;
	        if(array.Type() == SOA8) kernel_pointer = &kernel_PHeatBath_evenodd<false, SOA8, Real>;
			#else
	        kernel_pointer = &kernel_PHeatBath_evenodd<false, SOA, Real>;	
			#endif
	    }
	}
	if(kernel_pointer == NULL) errorCULQCD("No kernel HeatBath function exist for this gauge array...");
}

template <class Real> 
void HeatBath<Real>::apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      kernel_pointer<<<tp.grid,tp.block, 0, stream>>>(array.GetPtr(), randstates.state, parity, dir);
}
template <class Real> 
void HeatBath<Real>::Run(const cudaStream_t &stream){
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
template <class Real> 
void HeatBath<Real>::Run(){
	Run(0);
}
template <class Real> 
double HeatBath<Real>::time(){
	return timesec;
}

template <class Real> 
void HeatBath<Real>::stat(){
	COUT << "HeatBath:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
}


template <class Real> 
long long HeatBath<Real>::flop() const {
    //minumum flop for heatbath
	//NEEEDDDDD TO RECOUNT THIS PART!!!!!!!!!!!!!!!!!!!!!!!!! 
	#if (NCOLORS == 3)
	long long stapleflop = 2268LL ;
	long long phbflop = 801LL;
	long long ThreadFlop_phb = (7LL * array.getNumFlop(true) + array.getNumFlop(false) + stapleflop + phbflop) * size;
	#else
	double phbflop = NCOLORS * NCOLORS * NCOLORS + (NCOLORS * ( NCOLORS - 1) / 2) * (46LL + 48LL+56LL * NCOLORS);
	double stapleflop = NCOLORS * NCOLORS * NCOLORS * 84LL ;
    long long ThreadFlop_phb = (stapleflop + phbflop) * size;
	#endif
	#ifdef MULTI_GPU
	return ThreadFlop_phb * numnodes();
	#else
	return ThreadFlop_phb;
	#endif
}
template <class Real> 
long long HeatBath<Real>::bytes() const { 
    #ifdef MULTI_GPU
    return (20LL * array.getNumParams() * sizeof(Real) + 2LL * sizeof(cuRNGState)) * size * numnodes();
	#else
    return (20LL * array.getNumParams() * sizeof(Real) + 2LL * sizeof(cuRNGState))  * size;	
	#endif
}

template <class Real> 
double HeatBath<Real>::flops(){
	return ((double)flop() * 8 * 1.0e-9) / timesec;
}
template <class Real> 
double HeatBath<Real>::bandwidth(){
	return (double)bytes() * 8 / (timesec * (double)(1 << 30));
}






template class HeatBath<float>;
template class HeatBath<double>;

}
