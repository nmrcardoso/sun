#pragma once
#include<mop/wloop_arg.h>
namespace CULQCD{
template<bool UseTex, class Real, ArrayType atype> 
DEVICE void N1( WLOPArg<Real> arg,int id, int lx, int muvolume, msun line_left, msun line_right, int gfoffset1, int *pos){
//    if(id==0)printf("the symmetry inside N1 is %d\n",arg.symmetry);
	msun mop=msun::zero();
	int dir1 = arg.mu;
	int dmu[2]={(dir1+1)%3,(dir1+2)%3};
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//  _____                                        //
	//  |   |____ line                               //
	//  svec[i]                                      //
	///////////////////////////////////////////////////
	{
	int ids[2]={id, id};
	int halfline = arg.radius/2;
	for(int il = 0; il < 2; il++){
		msun link0=msun::identity();
		for(int ix=0;ix<lx;ix++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il]* DEVPARAMS::Volume, DEVPARAMS::size);
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], 1);
		}
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		for(int ix=0; ix<lx; ix++){
		ids[il]=Index_4D_Neig_NM(ids[il],dmu[il],-1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		if(arg.symmetry==0 || arg.symmetry==2) mop += link0 * line_right;//sigma_g_plus sigma_u_plus
		if(arg.symmetry==4 || arg.symmetry==5){//pi_g pi_u
		if(il==0)mop += link0 * line_right;
		if(il==1)mop += timesI(link0 * line_right);
		}

		if(arg.symmetry==6 || arg.symmetry==7){//delta_g , delta_u
		if(il==0)mop += link0 * line_right;
		if(il==1)mop -= link0 * line_right;
		}
	}
	
	//2 comp, in downway
	ids[0] = id;
	ids[1] = id;
	for(int il = 0; il < 2; il++){
		msun link0=msun::identity();
		for(int ix=0;ix<lx;ix++){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		for(int ix=0;ix<lx;ix++){
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids[il]=Index_4D_Neig_NM(ids[il], dmu[il], 1);
		}
		if(arg.symmetry==0 || arg.symmetry==2)mop += link0 * line_right;//sigma_g_plus, sigma_u_plus
		if(arg.symmetry==4 ||arg.symmetry==5){//pi_g, pi_u
			if(il==0) mop-=link0 * line_right;//pi_u
			if(il==1)mop-=timesI(link0 * line_right);//pi_u
		}

		if(arg.symmetry==6 || arg.symmetry==7){//delta_g, delta_u
			if(il==0)mop+=link0 * line_right;
			if(il==1)mop-=link0 * line_right;
		}
		}
	}//block for left staple
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//      _____                                    //
	//  ____|   | svec[i]                            //
	//   line                                        //
	///////////////////////////////////////////////////
	{
	int halfline = (arg.radius + 1) / 2;
	int s = Index_4D_Neig_NM(id, dir1, halfline);
	int ids[2];
	//2 comp, in upway
	
	for(int il = 0; il < 2; il++){
		ids[0] = s;
		ids[1] = s;
		msun link0=msun::identity();
		for(int ix=0;ix<lx;ix++){
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids[il]=Index_4D_Neig_NM(ids[il], dmu[il],1);
		}
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		for(int ix=0;ix<lx;ix++){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il],-1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il]+ dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		if (arg.symmetry==0) mop += line_left * link0;//sigma_g_plus
		if (arg.symmetry==2) mop -= line_left * link0;//sigma_u_plus
		if(arg.symmetry==5){
			if (il==0) mop += line_left * link0;//pi_u
			if (il==1) mop += timesI(line_left * link0);
		}
		if(arg.symmetry==4){
			if (il==0) mop -= line_left * link0;//pi_g
			if (il==1) mop -= timesI(line_left * link0);
		}
		if(arg.symmetry==6){//delta_g
			if(il==0)mop+=line_left * link0;
			if(il==1)mop-=line_left * link0;
		}
		if(arg.symmetry==7){//delta_u
			if(il==0)mop-=line_left * link0;
			if(il==1)mop+=line_left * link0;
		}
	}

	//2 comp, in downway
	for(int il = 0; il < 2; il++){
		ids[0] = s;
		ids[1] = s;
		msun link0=msun::identity();
		for(int ix=0;ix<lx;ix++){
			ids[il]=Index_4D_Neig_NM(ids[il], dmu[il],-1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = Index_4D_Neig_NM(ids[il], dir1, 1);
		}
		for(int ix=0;ix<lx;ix++){
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids[il]=Index_4D_Neig_NM(ids[il], dmu[il],1);
		}
		if(arg.symmetry==0)mop += line_left * link0;//sigma_g_plus
		if(arg.symmetry==2)mop -= line_left * link0;//sigma_u_plus
		if(arg.symmetry==5){
		if(il==0)mop -= line_left * link0;//pi_u
		if(il==1)mop -= timesI(line_left * link0);//pi_u
		}
		if(arg.symmetry==4){
		if(il==0)mop += line_left * link0;//pi_g
		if(il==1)mop += timesI(line_left * link0);//pi_g
		}
		if(arg.symmetry==6){//delta_g
			if(il==0)mop+=line_left * link0;
			if(il==1)mop-=line_left * link0;
		}
		if(arg.symmetry==7){//delta_u
			if(il==0)mop-=line_left * link0;
			if(il==1)mop+=line_left * link0;
		}
	}
	}//end of block for right staple
	mop /= (2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + (*pos) * DEVPARAMS::Volume, gfoffset1);
	(*pos)++;
}
}
