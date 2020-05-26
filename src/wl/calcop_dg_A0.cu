
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
#include <staple.h>
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

using namespace std;



namespace CULQCD{


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
//     |      |lx                               //
//     |      |   MO=More operator              //
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
//     /      /         Alireza                 //
//    /ly    /          MO1                     //
//    |      |                                  //
//    |lx    |                                  //
//////////////////////////////////////////////////
//another version of MO1
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO1(WLOPArg<Real> arg,int id, int lx, int ly, int muvolume){
	msun mop = msun::zero();
	{
	int dir1 = arg.mu;
	int dmu[2], dmu1[2]; 
	dmu[0]=(dir1+1)%3;
	dmu[1]=(dir1+2)%3;
	dmu1[0]=dmu[1];
	dmu1[1]=dmu[0];
	int ids[2];// these are just two variables update by moving on the path
	msun right, left;
	for(int il=0;il<2;il++){
		right=msun::identity();
		left=msun::identity();
		ids[0]=id;
		ids[1]=Index_4D_Neig_NM(id, dir1, arg.radius);
		//1st up
		for(int ix=0;ix<lx;ix++){
			left*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size );
			right*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[1]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			ids[1]=Index_4D_Neig_NM(ids[1], dmu[il], 1);
		}
		//2nd part
		{
		//2nd forward
		msun link=msun::identity();
		for(int iy=0;iy<ly;iy++){
			link*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size );
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il], 1);
		}
		for(int ir=0;ir<arg.radius;ir++){
			link*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[0]+muvolume, DEVPARAMS::size );
			ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
		}
		for(int iy=0;iy<ly;iy++){
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il],-1);
			link*=GAUGE_LOAD_DAGGER<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size );
		}
		mop+=left*link*right.dagger();
		//2nd backward
		//refreshing variables
		link=msun::identity();
		ids[0]=Index_4D_Neig_NM(id, dmu[il], lx);//first(up)
		for(int iy=0;iy<ly;iy++){
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il], -1);
			link*=GAUGE_LOAD_DAGGER<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size );
		}
		for(int ir=0;ir<arg.radius;ir++){
			link*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[0]+muvolume, DEVPARAMS::size );
			ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
		}
		for(int iy=0;iy<ly;iy++){
			link*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size );
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il],1);
		}
		mop+=left*link*right.dagger();
		}//end of block for 2nd part(forward)
		//1st part down
		//refreshing variables
		right=msun::identity();
		left=msun::identity();
		ids[0]=id;
		ids[1]=Index_4D_Neig_NM(id, dir1, arg.radius);
		//1st down
		for(int ix=0;ix<lx;ix++){
			ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
			ids[1]=Index_4D_Neig_NM(ids[1], dmu[il],-1);
			left*=GAUGE_LOAD_DAGGER<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size );
			right*=GAUGE_LOAD_DAGGER<UseTex,atype,Real>(arg.gaugefield,ids[1]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		//2nd part
		{
		//2nd forward
		msun link=msun::identity();
		for(int iy=0;iy<ly;iy++){
			link*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size );
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il], 1);
		}
		for(int ir=0;ir<arg.radius;ir++){
			link*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[0]+muvolume, DEVPARAMS::size );
			ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
		}
		for(int iy=0;iy<ly;iy++){
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il],-1);
			link*=GAUGE_LOAD_DAGGER<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size );
		}
		mop+=left*link*right.dagger();
		//2nd backward
		//refreshing variables
		link=msun::identity();
		ids[0]=Index_4D_Neig_NM(id, dmu[il],-lx);//1st down
		for(int iy=0;iy<ly;iy++){
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il], -1);
			link*=GAUGE_LOAD_DAGGER<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size );
		}
		for(int ir=0;ir<arg.radius;ir++){
			link*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[0]+muvolume, DEVPARAMS::size );
			ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
		}
		for(int iy=0;iy<ly;iy++){
			link*=GAUGE_LOAD<UseTex,atype,Real>(arg.gaugefield,ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size );
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il],1);
		}
		mop+=left*link*right.dagger();
		}//end of block 2nd part(backward)
	}//end of for(il)
	}//end of block
	return mop;
}
///////////////////////////////////////
//      _________                    //
//      |lx2    |  Alireza           //
//      /       /ly  N=16            //
//      |lx1    |                    //
//      MO2                          //
///////////////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO2(WLOPArg<Real> arg,int id, int lx1, int ly, int lx2, int muvolume){
	msun mop = msun::zero();
	{
	int dir1 = arg.mu;
	int dmu[2], dmu1[2]; 
	dmu[0]=(dir1+1)%3;
	dmu[1]=(dir1+2)%3;
	dmu1[0]=dmu[1];
	dmu1[1]=dmu[0];
	int ids[2];//ids[0] for the left side of the path, and ids[1] for right side
	msun link0, right, left;
	for(int il=0;il<2;il++){
		ids[0]=id;
		ids[1]=Index_4D_Neig_NM(id, dir1, arg.radius);
		left=msun::identity();
		right=msun::identity();
		//1st up
		for(int ix1=0;ix1<lx1;ix1++){
			left*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			right*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[1]=Index_4D_Neig_NM(ids[1], dmu[il], 1);
		}
		//2nd forward
		{
			msun leftp=msun::identity();//leftprim
			msun rightp=msun::identity();//rightprim
			for(int iy=0;iy<ly;iy++){
				leftp*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il], 1);
				rightp*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[1]=Index_4D_Neig_NM(ids[1], dmu1[il], 1);
			}
		// 3p up
			link0=msun::identity(); 
			for(int ix2=0;ix2<lx2;ix2++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			}
			for(int ir=0;ir<arg.radius;ir++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+muvolume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
			}
			for(int ix2=0;ix2<lx2;ix2++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
				link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
			mop+=left*leftp*link0*(right*rightp).dagger();
		// 3p down
		//refreshing variables
			link0=msun::identity();
			ids[0]=Index_4D_Neig_NM(id, dmu[il], lx1, dmu1[il],ly);//1st up, 2nd forward
			for(int ix2=0;ix2<lx2;ix2++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
				link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
			for(int ir=0;ir<arg.radius;ir++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+muvolume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
			}
			for(int ix2=0;ix2<lx2;ix2++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			}
			mop+=left*leftp*link0*(right*rightp).dagger();
		}
		// end of 2nd part's block
		///////////////////
		//2nd backward
		//refreshing coordinates
		ids[0]=Index_4D_Neig_NM(id,dmu[il], lx1);// 1st up
		ids[1]=Index_4D_Neig_NM(id, dir1, arg.radius, dmu[il], lx1);//right side coordinates
		{
			msun leftp=msun::identity();
			msun rightp=msun::identity();
			for(int iy=0;iy<ly;iy++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il],-1);
				leftp*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[1]=Index_4D_Neig_NM(ids[1], dmu1[il],-1);
				rightp*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
		// 3p up
			link0=msun::identity(); 
			for(int ix2=0;ix2<lx2;ix2++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			}
			for(int ir=0;ir<arg.radius;ir++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+muvolume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
			}
			for(int ix2=0;ix2<lx2;ix2++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
				link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
			mop+=left*leftp*link0*(right*rightp).dagger();
		// 3p down
		// refreshing variables
			link0=msun::identity();
			ids[0]=Index_4D_Neig_NM(id, dmu[il],lx1, dmu1[il],-ly);//2nd part is backward
			for(int ix2=0;ix2<lx2;ix2++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
				link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
			for(int ir=0;ir<arg.radius;ir++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+muvolume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
			}
			for(int ix2=0;ix2<lx2;ix2++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			}
			mop+=left*leftp*link0*(right*rightp).dagger();
		}
// end of 2nd block
//////////////////////////////////
//      1 part downway          //
//////////////////////////////////
/*refreshing coordinates*/
		ids[0]=id;
		ids[1]=Index_4D_Neig_NM(id, dir1, arg.radius);
		left=msun::identity();
		right=msun::identity();
		for(int ix1=0;ix1<lx1;ix1++){
			ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
			left*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[1]=Index_4D_Neig_NM(ids[1], dmu[il],-1);
			right*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		//second forward
		{
			msun leftp=msun::identity();
			msun rightp=msun::identity();
			for(int iy=0;iy<ly;iy++){
				leftp*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il], 1);
				rightp*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[1]=Index_4D_Neig_NM(ids[1], dmu1[il], 1);
			}
		// 3p up
			link0=msun::identity(); 
			for(int ix2=0;ix2<lx2;ix2++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			}
			for(int ir=0;ir<arg.radius;ir++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+muvolume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
			}
			for(int ix2=0;ix2<lx2;ix2++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
				link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
			mop+=left*leftp*link0*(right*rightp).dagger();
		// 3p down
			link0=msun::identity();
			ids[0]=Index_4D_Neig_NM(id, dmu[il], -lx1, dmu1[il],ly);//1st down, 2nd forward
			for(int ix2=0;ix2<lx2;ix2++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
				link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
			for(int ir=0;ir<arg.radius;ir++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+muvolume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
			}
			for(int ix2=0;ix2<lx2;ix2++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			}
			mop+=left*leftp*link0*(right*rightp).dagger();
		}
		// end of first block
		// refreshing coordinates
		ids[0]=Index_4D_Neig_NM(id, dmu[il], -lx1);//1st down
		ids[1]=Index_4D_Neig_NM(id, dir1, arg.radius, dmu[il], -lx1);
		// 2nd backward
		{
			msun leftp=msun::identity();
			msun rightp=msun::identity();
			for(int iy=0;iy<ly;iy++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il],-1);
				ids[1]=Index_4D_Neig_NM(ids[1], dmu1[il],-1);
				leftp*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				rightp*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
		// 3p up
			link0=msun::identity(); 
			for(int ix2=0;ix2<lx2;ix2++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			}
			for(int ir=0;ir<arg.radius;ir++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+muvolume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
			}
			for(int ix2=0;ix2<lx2;ix2++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
				link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
			mop+=left*leftp*link0*(right*rightp).dagger();
		// 3p down
			link0=msun::identity();
			ids[0]=Index_4D_Neig_NM(id, dmu[il],-lx1, dmu1[il],-ly);//1st down, 2nd backward 
			for(int ix2=0;ix2<lx2;ix2++){
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il],-1);
				link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			}
			for(int ir=0;ir<arg.radius;ir++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+muvolume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dir1, 1);
			}
			for(int ix2=0;ix2<lx2;ix2++){
				link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
				ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			}
			mop+=left*leftp*link0*(right*rightp).dagger();
		}
	}// end of for(il)
	}//end of the block, all other variable are inside this block expect mop
	return mop;
}





//////////////////////

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
	for(int r = 0; r < arg.radius; r++){
		int idx = Index_4D_Neig_NM(x, arg.mu, r);
		link *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idx + muvolume, DEVPARAMS::size);
	}
	if(arg.opN == 1){
        GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id, DEVPARAMS::Volume);
        return;
    }
	else GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id, gfoffset1);





	//COMMON PART
	int halfline = (arg.radius + 1) / 2;
	msun line_left = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + muvolume, DEVPARAMS::size);
	for(int ir = 1; ir < halfline; ++ir) 
		line_left *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, arg.mu, ir) + muvolume, DEVPARAMS::size);

	halfline = arg.radius/2;
	msun line_right = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, arg.mu, halfline) + muvolume, DEVPARAMS::size);
	for(int ir = halfline + 1; ir < arg.radius; ++ir)
		line_right *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, arg.mu, ir) + muvolume, DEVPARAMS::size);



{
	msun mop = msun::zero();
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//  _____                                        //
	//  |   |____ line                               //
	//  svec[i]                                      //
	///////////////////////////////////////////////////
	int dir1 = arg.mu;
	int dir2 = (dir1+1)%3;
	int dir3 = (dir1+2)%3;
	int ids[2]; int dmu[2];
	ids[0] = Index_4D_Neig_NM(x, dir2, 1);
	ids[1] = Index_4D_Neig_NM(x, dir3, 1);
	dmu[0] = dir2 * DEVPARAMS::Volume;
	dmu[1] = dir3 * DEVPARAMS::Volume;
	halfline = arg.radius/2;
	//2 comp, in upway
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + dmu[il], DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, dir1, halfline) + dmu[il], DEVPARAMS::size);
		mop += link0 * line_right;
	}
	ids[0] = Index_4D_Neig_NM(x, dir2, -1);
	ids[1] = Index_4D_Neig_NM(x, dir3, -1);
	//2 comp, in downway
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		mop += link0 * line_right;
	}
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//      _____                                    //
	//  ____|   | svec[i]                            //
	//   line                                        //
	///////////////////////////////////////////////////
	halfline = (arg.radius + 1) / 2;
	int s2 = Index_4D_Neig_NM(x, dir1, halfline);
	ids[0] = Index_4D_Neig_NM(s2, dir2, 1);
	ids[1] = Index_4D_Neig_NM(s2, dir3, 1);
	//2 comp, in upway
	for(int il = 0; il < 2; il++){		
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, s2 + dmu[il], DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, dir1, arg.radius) + dmu[il], DEVPARAMS::size);
		mop += line_left * link0;
	}
	ids[0] = Index_4D_Neig_NM(s2, dir2, -1);
	ids[1] = Index_4D_Neig_NM(s2, dir3, -1);
	//2 comp, in downway
	for(int il = 0; il < 2; il++){		
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		mop += line_left * link0;   
	}
    mop /= (2.0*sqrt(2.0));
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + DEVPARAMS::Volume, gfoffset1);
}





{
	msun mop = msun::zero();
  //for(int np = 0; np < Nbase; np++) mop[np]=msun::zero();
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
    //   ____                                        //
	//  /   /                                        //
	//  |   |____ line                               //
	//  svec[i]                                      //
	///////////////////////////////////////////////////
	int dir1 = arg.mu;
	int dir2 = (dir1+1)%3;
	int dir3 = (dir1+2)%3;
	int ids[2]; int dmu[2]; int dmu1[2];
	ids[0] = Index_4D_Neig_NM(x, dir2, 1);
	ids[1] = Index_4D_Neig_NM(x, dir3, 1);
	dmu[0] = dir2;
	dmu[1] = dir3;
	dmu1[0] = dir3;
	dmu1[1] = dir2;
	halfline = arg.radius/2;
	//2 comp, 1st upway 2nd foward  0/1
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(id, dmu[il], 1, dmu1[il], 1);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
    ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], -1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
 		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, dir1, halfline) + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		mop += link0 * line_right;
	}
	//2 comp, 1st upway 2nd backward  2/3
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(id, dmu[il], 1, dmu1[il], -1);
    link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);

		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
 		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, dir1, halfline) + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		mop += link0 * line_right;
	}

	ids[0] = Index_4D_Neig_NM(x, dir2, -1);
	ids[1] = Index_4D_Neig_NM(x, dir3, -1);
	//2 comp, 1st downway 2nd foward 4/5
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], 1);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
    ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], -1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		mop += link0 * line_right;
	}

	ids[0] = Index_4D_Neig_NM(x, dir2, -1);
	ids[1] = Index_4D_Neig_NM(x, dir3, -1);
	//2 comp, 1st downway 2nd backward 6/7
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], -1);
    link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], 1);
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		mop += link0 * line_right;
	}








	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
    //       ____                                    //
	//      /   /                                    //
	//  ____|   | svec[i]                            //
	//   line                                        //
	///////////////////////////////////////////////////
	halfline = (arg.radius + 1) / 2;
	int s2 = Index_4D_Neig_NM(x, dir1, halfline);
	ids[0] = Index_4D_Neig_NM(s2, dir2, 1);
	ids[1] = Index_4D_Neig_NM(s2, dir3, 1);
    //2 comp, 1st upway 2nd foward 8/9
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, s2 + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(s2, dmu[il], 1);
    link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], 1);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
    ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], -1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(ids[il], dmu[il], -1);
 		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		mop += line_left * link0;
	}
	//2 comp, 1st upway 2nd backward 10/11
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, s2 + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(s2, dmu[il], 1, dmu1[il], -1);
    link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = Index_4D_Neig_NM(ids[il], dmu[il], -1, dmu1[il], 1);
 		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		mop += line_left * link0;
	}
	ids[0] = Index_4D_Neig_NM(s2, dir2, -1);
	ids[1] = Index_4D_Neig_NM(s2, dir3, -1);
	//2 comp, 1st downway 2nd foward 12/13
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
        link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
        ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], 1);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
        ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], -1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		mop += line_left * link0;
	}
	ids[0] = Index_4D_Neig_NM(s2, dir2, -1);
	ids[1] = Index_4D_Neig_NM(s2, dir3, -1);
	//2 comp, 1st downway 2nd backward 14/15
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
        ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
        ids[il] = Index_4D_Neig_NM(ids[il], dmu1[il], 1);
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		mop += line_left * link0;
	}
    mop *= 0.25;
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + 2 * DEVPARAMS::Volume, gfoffset1);
}





{
	msun mop = msun::zero();
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//  _____                                        //
	//  |___|____ line                               //
	//  svec[i]                                      //
	///////////////////////////////////////////////////
	int dir1 = arg.mu;
	int dir2 = (dir1+1)%3;
	int dir3 = (dir1+2)%3;
	int ids[2]; int dmu[2];
	ids[0] = Index_4D_Neig_NM(x, dir2, 1);
	ids[1] = Index_4D_Neig_NM(x, dir3, 1);
	dmu[0] = dir2 * DEVPARAMS::Volume;
	dmu[1] = dir3 * DEVPARAMS::Volume;
	halfline = arg.radius/2;
    msun line0= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + muvolume, DEVPARAMS::size);
	for(int ir = 1; ir < halfline; ++ir) 
		line0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, dir1, ir) + muvolume, DEVPARAMS::size);
	//2 comp, in upway
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + dmu[il], DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, dir1, halfline) + dmu[il], DEVPARAMS::size);
		msun link = line0 * link0.dagger() * line0 * line_right;
		mop += link;
	}
	ids[0] = Index_4D_Neig_NM(x, dir2, -1);
	ids[1] = Index_4D_Neig_NM(x, dir3, -1);
	//2 comp, in downway
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		msun link = line0 * link0.dagger() * line0 * line_right;
		mop += link;
	}
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//      _____                                    //
	//  ____|___| svec[i]                            //
	//   line                                        //
	///////////////////////////////////////////////////
	halfline = (arg.radius + 1) / 2;
	int s2 = Index_4D_Neig_NM(x, dir1, halfline);
	ids[0] = Index_4D_Neig_NM(s2, dir2, 1);
	ids[1] = Index_4D_Neig_NM(s2, dir3, 1);
    line0= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, dir1, halfline) + muvolume, DEVPARAMS::size);
	for(int ir = halfline + 1; ir < arg.radius; ++ir) 
		line0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, dir1, ir) + muvolume, DEVPARAMS::size);
	//2 comp, in upway
	for(int il = 0; il < 2; il++){		
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, s2 + dmu[il], DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, dir1, arg.radius) + dmu[il], DEVPARAMS::size);
		msun link = line_left * line0 * link0.dagger() * line0;
		mop += link;
	}
	ids[0] = Index_4D_Neig_NM(s2, dir2, -1);
	ids[1] = Index_4D_Neig_NM(s2, dir3, -1);
	//2 comp, in downway
	for(int il = 0; il < 2; il++){		
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		msun link = line_left * line0 * link0.dagger() * line0;
		mop += link;
	}
    mop/=(2.0*sqrt(2.0));
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + 3 * DEVPARAMS::Volume, gfoffset1);
}
//template<bool UseTex, class Real, ArrayType atype> 
//DEVICE msun* MO0(WLOPArg<Real> arg,int id, int lx, int muvolume)
//DEVICE msun* MO1(WLOPArg<Real> arg,int id, int lx, int ly, int muvolume)
///////////////////////////////////////////////
//      new operator section                 //
///////////////////////////////////////////////
{
int index=3;
// this part is for set MO0
{ 
int set0_l[4]={1, 2, 3, 4};
int n=sizeof(set0_l)/sizeof(int);
for(uint i=0;i<n;i++){
	{
	index++;
	msun mop=MO0<UseTex, Real, atype>(arg, id, set0_l[i],  muvolume);
	mop/=(2.0);
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index * DEVPARAMS::Volume, gfoffset1);
	}
}
} 
///////////////////////////
// this part is for set2 MO1
{
int set1_lx[9]={1,1,2,2,1,3,2,3,3};
int set1_ly[9]={1,2,1,2,3,1,3,2,3};
int n=sizeof(set1_lx)/sizeof(int);
for(uint i=0;i< n;i++){
	index++;
	{
	msun mop=MO1<UseTex, Real, atype>(arg, id, set1_lx[i], set1_ly[i],  muvolume);
	mop/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index * DEVPARAMS::Volume, gfoffset1);
	}
}
}
//////////////////////////////
// this part is for set3 MO2
{
int set3_l1x[9]={1,1,2,1,1,2,1,3,2};
int set3_ly[9]= {1,1,1,2,3,1,1,1,2};
int set3_l2x[9]={1,2,1,1,1,2,3,1,2};
int n=sizeof(set3_l1x)/sizeof(int);
for(uint i=0;i<n;i++){
	index++;
	{
	msun mop=MO2<UseTex, Real, atype>(arg, id, set3_l1x[i], set3_ly[i], set3_l2x[i],  muvolume); 
	mop*=0.25;
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index* DEVPARAMS::Volume, gfoffset1);
	}
}
}
/////////////////////////////
}
//end of block for More Operator

}
//////////////////////////////////////
// end of kernal for calculation of //
// operators                        //
//////////////////////////////////////
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

