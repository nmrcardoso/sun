#pragma once
#include<mop/wloop_arg.h>
namespace CULQCD{
//////////////////////////////////////////////////S
//                 Alireza                      //
//      ______                                  //
//     |      |lx                               //
//     |      |   MO=More operator              //
//                                              //
//////////////////////////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE void MO0(WLOPArg<Real> arg,int id, int lx, int muvolume, int gfoffset1, int *pos){
//    if(id==0)printf("the symmetry inside mo0 is %d\n",arg.symmetry);
	msun mop = msun::zero();
	int dir1 = arg.mu;
	int dmu[2];
	dmu[0]=(dir1+1)%3;
	dmu[1]=(dir1+2)%3;
	//0 comp 1st upway
	for(int il=0;il<2;il++){
		int ids=id;
		msun link0=msun::identity();
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
		if(arg.symmetry==0)mop+=link0; //sigma_g_plus
		if(arg.symmetry==6){//delta_g
			if(il==0)mop+=link0; 
			if(il==1)mop-=link0;
			}
		if(arg.symmetry==5){
			if(il==0)mop+=link0;
			if(il==1)mop+=timesI(link0);
		}
			}
	//1 comp 1st downway
	for(int il=0; il<2; il++){
		int ids=id;
		msun link0=msun::identity();
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
		if(arg.symmetry==0)mop+=link0;//sigma_g_plus
		if(arg.symmetry==6){//delta_g
		if(il==0)mop+=link0; 
		if(il==1)mop-=link0;
		}
		if(arg.symmetry==5){//Pi_u
			if(il==0) mop-=link0;
			if(il==1) mop-=timesI(link0);
		}
		}
	mop/=(2.0);
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + (*pos) * DEVPARAMS::Volume, gfoffset1);
	(*pos)++;
}

}
