#pragma once
#include<mop/wloop_arg.h>
namespace CULQCD{
//////////////////////////////////////////////////
//      _______                                 //
//     /      /         Alireza                 //
//    /ly    /          MO1                     //
//    |      |                                  //
//    |lx    |                                  //
//////////////////////////////////////////////////
//another version of MO1
template<bool UseTex, class Real, ArrayType atype> 
DEVICE void MO1(WLOPArg<Real> arg,int id, int lx, int ly, int muvolume, int gfoffset1, int *pos, int offset){
//    if(id==0)printf("the symmetry inside mo1 is %d\n",arg.symmetry);
	msun mop[2] ={ msun::zero(), msun::zero()};
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
		msun res=left*link*right.dagger();
		if (arg.symmetry==0) mop[0]+=res;//sigma_g_plus
		if (arg.symmetry==1){//sigma_g_minus 
		if (il==0) mop[0]+=res; //Vx(l1, l2) 
		if (il==1) mop[0]-=res; //Vy(l1, l2) 
		}
		//notice, for pi_u only one of forward or backward is enough to satisfy
		//the symmetry // shouls turn off
		if(arg.symmetry==5){//pi_u
		if(il==0) {
		mop[0]+=res;
		mop[1]+=res;
		}
		if(il==1){
			mop[0]+=timesI(res);//
			mop[1]-=timesI(res);//
		}}
		//notice that the negative direction can be off and just rotate clockwise to satisfy symmetry
		if(arg.symmetry==6){ //delta_g,
		if(il==0){
			mop[0]+=res;
			mop[1]+=res;
		} //Vx(l1, l2) on
		if(il==1){
			mop[0]-=res; //Vy(l1, l2)
			mop[1]+=res; //Vy(l1, l2) 
		}}
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
		res=left*link*right.dagger();
		 if (arg.symmetry==0) mop[0]+=res;//sigma_g_plus
		 if (arg.symmetry==1){//sigma_g_minus
		 	if(il==1)mop[0]+=res;
		 	if(il==0)mop[0]-=res;
		 } 
		if(arg.symmetry==5){//pi_u
		if(il==1){
			mop[0]+=timesI(res);
			mop[1]+=timesI(res);
		}
		if(il==0){
			mop[0]+=res;// Vx(l1, -l2)
			mop[1]-=res;//-Vx(l1, -l2) 
		}}
		if(arg.symmetry==6){//delta_g,
			if(il==1) {
				mop[0]-=res; //Vy(l1,-l2) on
				mop[1]-=res; //Vy(l1,-l2) on
			}
			if(il==0){
				mop[0]+=res; //Vx(l1,-l2)
				mop[1]-=res; //Vx(l1,-l2) 
			}}
		}
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
		msun res=left*link*right.dagger();
		if (arg.symmetry==0) mop[0]+=res;//sigma_g_plus
		if (arg.symmetry==1){//sigma_g_minus
		if (il==1) mop[0]+=res;
		if (il==0) mop[0]-=res;
		}
		if(arg.symmetry==5){//pi_u
			if(il==1){
				mop[0]-=timesI(res); // Vy(-l1,l2) on
				mop[1]-=timesI(res); // Vy(-l1,l2) on
			}
			if(il==0){
				mop[0]-=res;// Vx(-l1, l2)
				mop[1]+=res;// Vx(-l1, l2) 
			}}
		if(arg.symmetry==6){ //delta_g,
			if(il==1){
				mop[0]-=res; //Vy(-l1,l2) on
				mop[1]-=res; //Vy(-l1,l2) on
			}
			if(il==0){
				mop[0]+=res; //Vx(-l1,l2) 
				mop[1]-=res; //-Vx(-l1,l2) 
			}
		}
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
		res=left*link*right.dagger();
		if (arg.symmetry==0) mop[0]+=res;//sigma_g_plus
		if (arg.symmetry==1){ //sigma_g_minus
		if(il==0) mop[0]+=res;
		if(il==1) mop[0]-=res;
		}
		if(arg.symmetry==5){//pi_u
			if(il==0){
				mop[0]-=res;// Vx(-l1,-l2) on
				mop[1]-=res;// Vx(-l1,-l2) on
				}
			if(il==1){
				mop[0]-=timesI(res); // Vy(-l1,-l2) 
				mop[1]+=timesI(res); // Vy(-l1,-l2) 
			}
			}
		if(arg.symmetry==6){ //delta_g,
			if(il==0){
				mop[0]+=res; //Vx(-l1,-l2) on
				mop[1]+=res; //Vx(-l1,-l2) on
			}
			if(il==1){
				mop[0]-=res; //-Vy(-l1,-l2) off
				mop[1]+=res; //+Vy(-l1,-l2) off
			}}
		}//end of block 2nd part(backward)
	}//end of for(il)
	}//end of block
	mop[0]/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[0], id + (*pos) * DEVPARAMS::Volume, gfoffset1);
	(*pos)++;
	if((arg.symmetry==5||arg.symmetry==6) & (offset!=0)){
		mop[1]/=(2.0*sqrt(2.0));
		GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[1], id + ((*pos)+offset) * DEVPARAMS::Volume, gfoffset1);
		(*pos)++;
	}
}

}
