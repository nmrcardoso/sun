
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


using namespace std;


namespace CULQCD{


template<class Real>
Sigma_g_plus<Real>::Sigma_g_plus(const int _Rmax, const int _Tmax) : Rmax(_Rmax), Tmax(_Tmax){
	opN = 21;//9;//15;
	totalOpN = opN * opN;
	fieldOp.Set( SOA, Device, false);
	fieldOp.Allocate(PARAMS::Volume * opN);
	wloop_size = totalOpN * (Tmax+1) * sizeof(Real);
	wloop = (Real*) dev_malloc( wloop_size );
	wloop_h = (Real*) safe_malloc( wloop_size );
}

template<class Real>
Sigma_g_plus<Real>::~Sigma_g_plus(){
	dev_free(wloop);
	host_free(wloop_h);
	fieldOp.Release();
}


template class Sigma_g_plus<float>;
template class Sigma_g_plus<double>;











template<class Real>
struct WLOPArg{
  complex *gaugefield;
  complex *fieldOp;
  int radius;
  int mu;
  int opN;
};
//////////////////////////////////////////////////
//                 Alireza                      //
//      ______                                  //
//     |      |                                 //
//     |      |                                 //
//                                              //
//////////////////////////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO0(WLOPArg<Real> arg,int id, int lx, int muvolume){
	msun mop = msun::zero();
	int dir1 = arg.mu;
	int dmu[2]; 
	dmu[0]=(dir1+1)%3;
	dmu[1]=(dir1+2)%3;
	int ids=id;
	msun link0;
	//0 comp 1st upway
	for(int il=0;il<2;il++){
		ids=id;
		link0=msun::identity();
		for(int ix=0; ix<lx;ix++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu[il], 1);
		}
		// dir1 comp
		for(int ir=0;ir<arg.radius;ir++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+muvolume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dir1, 1);
		}
		// downway comp
		for(int ix=0;ix<lx;ix++){
			ids=Index_4D_Neig_NM(ids, dmu[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		mop+=link0;
	}
	//1 comp 1st downway
	for(int il=0; il<2; il++){
		ids=id;
		link0=msun::identity();
		// downway comp
		for(int ix=0; ix<lx;ix++){
			ids=Index_4D_Neig_NM(ids, dmu[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		//dir1 comp
		for(int ir=0;ir<arg.radius;++ir){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+muvolume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dir1, 1);
		}
		//upway comp
		for(int ix=0; ix<lx; ix++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[il]*DEVPARAMS::Volume , DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu[il], 1);
		}
		mop+=link0;
	}
	return mop;
}
//////////////////////////////////////////////////
//      _______                                 //
//     /      /        Alireza                  //
//    /ly    /     there is a function that     //
//    |      |     does these type and return   //
//    |lx    |     the operator for.            //
// these parts can have different length        //
//      called MO0 stands for more opertore     //
//////////////////////////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO2D0(WLOPArg<Real> arg,int id, int lx, int ly, int muvolume){
	msun mop = msun::zero();
	{
	int dir1 = arg.mu;
	int dmu[2], dmu1[2]; 
	dmu[0]=(dir1+1)%3;
	dmu[1]=(dir1+2)%3;
	dmu1[0]=dmu[1];
	dmu1[1]=dmu[0];
	int ids[2];// these are just two variables update by moving on the path
	msun link0;
	//0 comp, 1st upway 2nd foward 
	for(int il=0; il<2;il++){
		link0=msun::identity();
		ids[il]=id;
		//upway
		for(int ix=0;ix<lx;ix++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], 1);
		}
		//forward
		for(int iy=0;iy<ly;iy++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dmu1[il], 1);
		}
		//dir1
		for(int ir=0; ir<arg.radius;ir++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield,ids[il]+muvolume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		//backward 
		for(int iy=0;iy<ly;++iy){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu1[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		//downway
		for(int ix=0;ix<lx;ix++){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu1[il], -1);//notice: it is iy, and ids[0]
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		mop+=link0;
	}
	//1 comp, 1st upway 2nd backward
	for(int il=0; il<2;il++){
		ids[il]=id;
		link0=msun::identity();
		// upway
		for(int ix=0;ix<lx;ix++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield,ids[il]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], 1);
		}
		//backward
		for(int iy=0; iy<ly;iy++){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu1[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		//dir1 part
		for(int ir=0;ir<arg.radius;++ir){
			link0*=GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield,ids[il]+muvolume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dir1 ,1);
		}
		//forward
		for(int iy=0;iy<ly;iy++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dmu1[il], 1);
		}
		//downway
		for(int ix=0;ix<lx;ix++){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		mop+=link0;
	}
	//2 comp, 1st downway 2nd forward
	for( int il=0;il<2;il++){
		link0=msun::identity();
		ids[il]=id;
		//downway
		for(int ix=0;ix<lx;ix++){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[il]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		// forward
		for( int iy=0;iy<ly;iy++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[il]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dmu1[il], 1);
		}
		// dir1 parts
		for(int ir=0;ir<arg.radius;ir++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[il]+muvolume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		//backward
		for(int iy=0;iy<ly;iy++){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu1[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[il]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		//upway
		for(int ix=0;ix<lx;ix++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[il]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], 1);
		}
		mop+=link0;
	}
	///////////////////////////////////////
	// 3 comp, 1st downway, 2nd backward.//
	///////////////////////////////////////
	for(int il=0;il<2;il++){
		ids[il]=id;
		link0=msun::identity();
		//downway
		for(int ix=0;ix<lx;ix++){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex , atype, Real>(arg.gaugefield, ids[il]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		//backward
		for(int iy=0;iy<ly;++iy){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu1[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[il]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		// dir1
		for(int ir=0; ir<arg.radius;ir++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[il]+muvolume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		//forward
		for(int iy=0; iy<ly;iy++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[il]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dmu1[il], 1);
		}
		//upway
		for(int ix=0; ix<lx; ix++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[il]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], 1);
		}
		mop+=link0;
	}
	}
	return mop;
} 

template<bool UseTex, class Real, ArrayType atype>
__global__ void kernel_CalcOPsF_A0_33(WLOPArg<Real> arg){
  	int id = INDEX1D();
	if(id >= DEVPARAMS::Volume) return;
	int x[4];
	Index_4D_NM(id, x);
	int muvolume = arg.mu * DEVPARAMS::Volume;
	//int gfoffset = arg.opN * DEVPARAMS::Volume;

	int gfoffset1 = arg.opN * DEVPARAMS::Volume;
	msun link = msun::identity();
	int pos = 0;
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
	
	for(int kk=1; kk<=20; ++kk){
		msun mop=MO0<UseTex, Real, atype>(arg, id, kk,  muvolume); 
		mop/=(2.0);
		GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
		pos++;
	}
	/*

	msun mop=MO0<UseTex, Real, atype>(arg, id, 1,  muvolume); //lx=1
	mop/=(2.0);
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	/////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 1, 1,  muvolume);//lx=1, ly=1
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	///////////////////////////////////////
	mop=MO0<UseTex, Real, atype>(arg, id, 2,  muvolume);//lx=2
	mop/=(2.0);
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	//////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 2, 1,  muvolume); //lx=2, ly=1
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	//////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 2, 2,  muvolume); //lx=2, ly=2
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	///////////////////////////////////////
	mop=MO0<UseTex, Real, atype>(arg, id, 3,  muvolume);//lx=3
	mop/=(2.0);
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	/////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 3, 1,  muvolume);//lx=3, ly=1
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	// 8 operators
	
	/////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 3, 2,  muvolume);//lx=3, ly=2
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	/////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 3, 3,  muvolume);//lx=3, ly=3
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	// 10 operators
	
	
	
	///////////////////////////////////////
	mop=MO0<UseTex, Real, atype>(arg, id, 4,  muvolume);//lx=4
	mop/=(2.0);
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	/////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 4, 1,  muvolume);//lx=4, ly=1
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	
	/////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 4, 2,  muvolume);//lx=4, ly=2
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	/////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 4, 3,  muvolume);//lx=4, ly=3
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	/////////////////////////////////////
	mop=MO2D0<UseTex, Real, atype>(arg, id, 4, 4,  muvolume);//lx=4, ly=4
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + pos * DEVPARAMS::Volume, gfoffset1);
	pos++;
	
	
	//15 operators
	
	*/
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
void CalcWLOPs_A0(gauge array, Sigma_g_plus<Real> *arg, int radius, int mu){
  Timer mtime;
  mtime.start(); 
  WLOPArg<Real> argK;
	argK.gaugefield = array.GetPtr();
	argK.fieldOp = arg->fieldOp.GetPtr();
	argK.radius = radius;
	argK.mu = mu;
	argK.opN = arg->opN;

  
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
void CalcWLOPs_A0(gauge array, Sigma_g_plus<Real> *arg, int radius, int mu){
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(array.GetPtr(), true);
    CalcWLOPs_A0<true, Real>(array, arg, radius, mu);
  }
  else CalcWLOPs_A0<false, Real>(array, arg, radius, mu);
}


template void CalcWLOPs_A0<double>(gauged array, Sigma_g_plus<double> *arg, int radius, int mu);

}

