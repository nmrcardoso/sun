
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

#include <meas/wloopex.h>
#include<vector>
////// operators //////
#include<mop/wloop_arg.h>
#include<mop/opn1.cuh>
#include<mop/opn2.cuh>
#include<mop/opn3.cuh>
#include<mop/mo0.cuh>
#include<mop/mo1.cuh>
#include<mop/mo2.cuh>
#include<mop/mo3.cuh>
//#include<mop/mo4.cuh>
#include<mop/mo4_wag.cuh>
//////////////////////

using namespace std;


namespace CULQCD{
template<class Real>
symmetry_sector<Real>::symmetry_sector(const int _Rmax, const int _Tmax,const int _opN,  int _sys) : Rmax(_Rmax), Tmax(_Tmax),opN(_opN), symmetry(_sys){
	totalOpN = opN * opN; 
	fieldOp.Set( SOA, Device, false);
	fieldOp.Allocate(PARAMS::Volume * opN);
	wloop_size = totalOpN * (Tmax+1) * sizeof(complex);
	wloop = (complex*) dev_malloc( wloop_size );
	wloop_h = (complex*) safe_malloc( wloop_size );
}
template<class Real>
symmetry_sector<Real>::~symmetry_sector(){
	dev_free(wloop);
	host_free(wloop_h);
	fieldOp.Release();
	//printf("destructor of symmetry_sector called.\n");
}

template class symmetry_sector<float>;
template class symmetry_sector<double>;


template<bool UseTex, class Real, ArrayType atype>
__global__ void kernel_CalcOPsF_A0_33(WLOPArg<Real> arg){
	int id = INDEX1D();
	if(id >= DEVPARAMS::Volume) return;
	int x[4];
	Index_4D_NM(id, x);
	int muvolume = arg.mu * DEVPARAMS::Volume;
	int pos=0;
	int gfoffset1 = arg.opN * DEVPARAMS::Volume;
	/////////// direct wilson line ///////////////////
	if(arg.symmetry==0){
	msun link = msun::identity();
	for(int r = 0; r < arg.radius; r++){
		int idx = Index_4D_Neig_NM(x, arg.mu, r);
		link *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idx + muvolume, DEVPARAMS::size);
	}
	if(arg.opN == 1){
        GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id, DEVPARAMS::Volume);
        return;
    }
	else GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id, gfoffset1);
		pos++;
    }


    //sigma_g_plus=0    | sigma_g_minus=1   |sigma_u_plus=2     |sigma_u_minus=3
    //pi_g=4            |pi_u=5             |delta_g=6          |delta_u=7
    
	if(arg.symmetry==0||arg.symmetry==5|| arg.symmetry==6){
		#pragma unroll
		for(int l=1;l<=8;l++){
			//MO0(WLOPArg<Real> arg,int id, int lx, int muvolume, int gfoffset1, int *pos)
			MO0<UseTex, Real,atype>(arg ,id,l ,muvolume, gfoffset1, &pos);
		}
		}
		
	if (arg.symmetry==1|| arg.symmetry==6){
	    #pragma unroll
	    for (int l=1; l<=8;l++)
	        MO1<UseTex, Real, atype>( arg, id,  1, l,  muvolume,  gfoffset1, &pos, 8);
	}
	
//	if(arg.symmetry==3){
//        msun link = msun::identity();
//		for(int r = 0; r < arg.radius; r++){
//		int idx = Index_4D_Neig_NM(x, arg.mu, r);
//		link *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idx + muvolume, DEVPARAMS::size);
//		}
//		#pragma unroll
//		for (int l=1;l<=12;l++){
//		    MO4<UseTex, Real, atype>(arg, id,l,l,link ,gfoffset1, &pos);
//		   }
//	} this is from before wagner method
	if(arg.symmetry==3){
	    //these are base on wagner computation
	    int rl[13]={0,0,0,0,0,1,1,2,2,3,3,4,4};
            //rl=[0,1,2,3,4,5,6,7,8,9,10,11,12,13];
        int rr[13]={0,1,2,3,4,4,5,5,6,6,7,7,8};
        msun linkl = msun::identity();
        msun linkm = msun::identity();
        msun linkr = msun::identity();
		for(int r = 0; r < rl[arg.radius]; r++){
		int idx = Index_4D_Neig_NM(x, arg.mu, r);
		linkl *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idx + muvolume, DEVPARAMS::size);
		}
		for(int r =rl[arg.radius] ; r < rr[arg.radius]; r++){
		int idx = Index_4D_Neig_NM(x, arg.mu, r);
		linkm *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idx + muvolume, DEVPARAMS::size);
		}
		for(int r =rr[arg.radius] ; r < arg.radius; r++){
		int idx = Index_4D_Neig_NM(x, arg.mu, r);
		linkr *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idx + muvolume, DEVPARAMS::size);
		}
		#pragma unroll
		for (int l=1;l<=8;l++){
		    MO4<UseTex, Real, atype>(arg, id,l,l,linkl, linkm, linkr ,gfoffset1, &pos);
		   }
	}
	
	
	// pay attention to 3
	if(arg.symmetry==2||arg.symmetry==4||arg.symmetry==7||arg.symmetry==3){
	//#########################################################################
	int halfline = (arg.radius + 1) / 2;
	msun line_left = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + muvolume, DEVPARAMS::size);
	for(int ir = 1; ir < halfline; ++ir) 
		line_left *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, arg.mu, ir) + muvolume, DEVPARAMS::size);
	halfline = arg.radius/2;
	msun line_right = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, arg.mu, halfline) + muvolume, DEVPARAMS::size);
	for(int ir = halfline + 1; ir < arg.radius; ++ir)
		line_right *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, arg.mu, ir) + muvolume, DEVPARAMS::size);
	if(arg.symmetry==2||arg.symmetry==4||arg.symmetry==7){
	#pragma unroll
	for(int l=1;l<=12;l++)
		N1< UseTex, Real, atype>(arg, id, l, muvolume, line_left, line_right, gfoffset1, &pos);
	
    }
	if (arg.symmetry==3){
		#pragma unroll
		for(int l=1;l<=8;l++)
			N2< UseTex, Real, atype>( arg, id, 1,  l,  muvolume, line_left, line_right, gfoffset1, &pos, 0);// offset if it is not 0, it compute both symmetry for epsilon
    }// this has been removed for action 1
   }
    
    if (arg.symmetry==3){
    	msun link = msun::identity();
    	for(int r = 0; r < arg.radius; r++){
		int idx = Index_4D_Neig_NM(x, arg.mu, r);
		link *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idx + muvolume, DEVPARAMS::size);
		}
    	//MO3(WLOPArg<Real> arg,int id, int lx1, int ly,int lx2,msun link,int gfoffset1, int *pos)
    	MO3< UseTex, Real, atype>(arg,id, 1,  1, 1, link, gfoffset1, &pos);
    	MO3< UseTex, Real, atype>(arg,id, 1,  2, 1, link, gfoffset1, &pos);
    	MO3< UseTex, Real, atype>(arg,id, 2,  1, 1, link, gfoffset1, &pos);
    	MO3< UseTex, Real, atype>(arg,id, 2,  2, 1, link, gfoffset1, &pos);
    	MO3< UseTex, Real, atype>(arg,id, 2,  2, 2, link, gfoffset1, &pos);
    }

}


template <bool UseTex, class Real, ArrayType atype> 
class CalcOPsF_A0: Tunable{
private:
   WLOPArg<Real> arg;
	gauge array;
	gauge fieldOp;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer CalcOPsF_A0time;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  //CUDA_SAFE_CALL(cudaMemset(arg.fieldOp, 0, PARAMS::Volume * arg.opN * sizeof(msun)));
	fieldOp.Clean();
  	kernel_CalcOPsF_A0_33<UseTex, Real, atype><<<tp.grid,tp.block, 0, stream>>>(arg);
}
public:
   CalcOPsF_A0(WLOPArg<Real> arg, gauge array, gauge fieldOp): arg(arg), array(array), fieldOp(fieldOp){
	size = 1;
	for(int i=0;i<4;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;  
}
   ~CalcOPsF_A0(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    CalcOPsF_A0time.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
    //CUDA_SAFE_CALL(cudaMemcpy(chromofield, arg.field, 6 * arg.nx * arg.ny * sizeof(Real), cudaMemcpyDeviceToHost));
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    CalcOPsF_A0time.stop();
    timesec = CalcOPsF_A0time.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const {
	long long tmp = (arg.radius - arg.radius/2 - 2LL + 4LL * (arg.radius / 2 + 1LL)) * array.getNumFlop(true) + 4LL * array.getNumFlop(false) + (arg.radius - arg.radius/2 - 1LL + 4LL * arg.radius / 2 + 4LL) * 198LL;
	tmp *= 2LL;
	tmp += arg.radius * (array.getNumFlop(true) + 198LL) + array.getNumFlop(false);
	tmp *= PARAMS::Volume;
	return tmp;}
   long long bytes() const{ 
		long long tmp = (arg.radius - arg.radius/2 - 2LL + 4LL * (arg.radius / 2 + 2LL)) * array.getNumParams() * sizeof(Real);
		tmp *= 2LL;
		tmp += (arg.radius + 1LL) * array.getNumParams() * sizeof(Real);
		tmp *= PARAMS::Volume;
return tmp;}
   double time(){	return timesec;}
   void stat(){	COUT << "CalcOPsF_A0:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  void preTune() { }
  void postTune() {  }

};






template<bool UseTex, class Real>
void CalcWLOPs_A0(gauge array, symmetry_sector<Real> *arg, int radius, int mu){
  Timer mtime;
  mtime.start(); 
  WLOPArg<Real> argK;
	argK.gaugefield = array.GetPtr();
	argK.fieldOp = arg->fieldOp.GetPtr();
	argK.radius = radius;
	argK.mu = mu;
	argK.opN = arg->opN;
	argK.symmetry=arg->symmetry;

  
  if(array.Type() != SOA || arg->fieldOp.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true || arg->fieldOp.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
    
  CalcOPsF_A0<UseTex, Real, SOA> wl(argK, array, arg->fieldOp);
  wl.Run();
  if (getVerbosity() >= VERBOSE) wl.stat();
  CUDA_SAFE_DEVICE_SYNC( );
  mtime.stop();
  if (getVerbosity() >= VERBOSE) COUT << "Time CalcOPsF_A0:  " <<  mtime.getElapsedTimeInSec() << " s"  << endl;
}




template<class Real>
void CalcWLOPs_A0(gauge array, symmetry_sector<Real> *arg, int radius, int mu){
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(array.GetPtr(), true);
    CalcWLOPs_A0<true, Real>(array, arg, radius, mu);
  }
  else CalcWLOPs_A0<false, Real>(array, arg, radius, mu);
}


template void CalcWLOPs_A0<double>(gauged array, symmetry_sector<double> *arg, int radius, int mu);

}

