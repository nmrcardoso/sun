#pragma once
#include<mop/wloop_arg.h>
namespace CULQCD{
///////////////////////////////////////
//      _________                    //
//      |lx2    |  Alireza           //
//      /       /ly  N=16            //
//      |lx1    |                    //
//      MO2                          //
///////////////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE void MO2(WLOPArg<Real> arg,int id, int lx1, int ly, int lx2, int muvolume, int gfoffset1, int *pos, int offset){
//    if(id==0)printf("the symmetry inside mo2 is %d\n",arg.symmetry);
	msun mop[2] ={msun::zero(), msun::zero()};
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
		//          -------lx2----
		//          | ly
		//-----lx1---
		for(int ix1=0;ix1<lx1;ix1++){
			left*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[0]=Index_4D_Neig_NM(ids[0], dmu[il], 1);
			right*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[1]=Index_4D_Neig_NM(ids[1], dmu[il], 1);
		}
		//2nd forward
		
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
		msun res=left*leftp*link0*(right*rightp).dagger();
		if(arg.symmetry==0)mop[0]+=res;
		if(arg.symmetry==1){//sigma_g_minus
			if(il==0) mop[0]+=res;
			if(il==1) mop[0]-=res;
		}
		// notice for pi_u only one of forward or backward is engouh
		// to satisfy the symmetry
		if(arg.symmetry==5){//pi_u
			if(il==0){
				mop[0]+=res;
				mop[1]+=res;
			}
			if(il==1){
				mop[0]+=timesI(res); //Vy(l1, l2, l3)
				mop[1]-=timesI(res); //Vy(l1, l2, l3) 
			}}
		if(arg.symmetry==6){//delta_g
			if(il==0){
				mop[0]+=res; //Vx(l1, l2, l3) on
				mop[1]+=res; //Vx(l1, l2, l3) on
			}
			if(il==1){
				mop[0]-=res; //Vy(l1, l2, l3) off
				mop[1]+=res; //Vy(l1, l2, l3) off
				}
			}
		//2nd backward
		//refreshing coordinates
		// ---lx1->---
		//           |ly
		//           ---->-lx2------
		ids[0]=Index_4D_Neig_NM(id,dmu[il], lx1);// 1st up
		ids[1]=Index_4D_Neig_NM(id, dir1, arg.radius, dmu[il], lx1);//right side coordinates
		leftp=msun::identity();
		rightp=msun::identity();
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
		res=left*leftp*link0*(right*rightp).dagger();
		if(arg.symmetry==0)mop[0]+=res;
		if(arg.symmetry==1){//sigma_g_minus
			if(il==1) mop[0]+=res;
			if(il==0) mop[0]-=res;
		}
		if(arg.symmetry==5){//pi_u
			if(il==1){
				mop[0]+=timesI(res);
				mop[1]+=timesI(res);
			}//Vy(l1, -l2, l3) on
			if(il==0){
				mop[0]+=res; //Vx(l1, -l2, l3) off
				mop[1]-=res; 
			}
		}
		if(arg.symmetry==6){//delta_g
			if(il==1){
				mop[0]-=res; //Vy(l1,-l2, l3) on
				mop[1]-=res; //Vy(l1,-l2, l3) on
			}
			if(il==0){
				mop[0]+=res; //Vx(l1,-l2, l3) 0ff
				mop[1]-=res; //-Vx(l1,-l2, l3) 0ff
			}
		}
		//////////////////////////////////
		//      1 part downway          //
		//////////////////////////////////
		//-<--lx2----
		//          | ly
		//          --<--lx1----
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

		leftp=msun::identity();
		rightp=msun::identity();
		for(int iy=0;iy<ly;iy++){
			leftp*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il], 1);
			rightp*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids[1]=Index_4D_Neig_NM(ids[1], dmu1[il], 1);
		}
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
		res=left*leftp*link0*(right*rightp).dagger();
		if(arg.symmetry==0)mop[0]+=res;
		if(arg.symmetry==1){//sigma_g_minus
			if(il==1) mop[0]+=res;
			if(il==0) mop[0]-=res;
		}
		if(arg.symmetry==5){//pi_u
			if(il==1){
				mop[0]-=timesI(res);//Vy(-l1, l2,-l3 ) on
				mop[1]-=timesI(res);//Vy(-l1, l2,-l3 ) on
			}
			if(il==0){
				mop[0]-=res;//-Vx(-l1, l2,-l3 )
				mop[1]+=res;//+Vx(-l1, l2,-l3 ) 
			}
		}
		if(arg.symmetry==6){//delta_g
			if(il==1){
				mop[0]-=res; //Vy(-l1,l2,-l3) on
				mop[1]-=res; //Vy(-l1,l2,-l3) on
			}
			if(il==0){
				mop[0]+=res; //+Vx(-l1,l2,-l3) off
				mop[1]-=res; //-Vx(-l1,l2,-l3) off
			}
		}
		////////////////////////////////////////////
		//        --< lx1-<--
		//        | ly
		//-lx2-<---

		//second backward
		ids[0]=Index_4D_Neig_NM(id, dmu[il], -lx1);//1st down
		ids[1]=Index_4D_Neig_NM(id, dir1, arg.radius, dmu[il], -lx1);
		
		leftp=msun::identity();
		rightp=msun::identity();
		for(int iy=0;iy<ly;iy++){
			ids[0]=Index_4D_Neig_NM(ids[0], dmu1[il],-1);
			ids[1]=Index_4D_Neig_NM(ids[1], dmu1[il],-1);
			leftp*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			rightp*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu1[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
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
		res=left*leftp*link0*(right*rightp).dagger();
		if(arg.symmetry==0) mop[0]+=res;
		if(arg.symmetry==1){//sigma_g_minus
			if(il==0) mop[0]+=res;
			if(il==1) mop[0]-=res;
		}
		if(arg.symmetry==5){//pi_u
			if(il==0){
				mop[0]-=res;//Vx(-l1, -l2, -l3) on
				mop[1]-=res;//Vx(-l1, -l2, -l3) on
			}
			if(il==1){
				mop[0]-=timesI(res);//-Vy(-l1, -l2, -l3)
				mop[1]+=timesI(res);//-Vy(-l1, -l2, -l3) 
			}}
		if(arg.symmetry==6){//delta_g
			if(il==0){
				mop[0]+=res; //Vx(-l1,-l2,-l3) on
				mop[1]+=res; //Vx(-l1,-l2,-l3) on
			}
			if(il==1){
				mop[0]-=res; //Vy(-l1,-l2,-l3)
				mop[1]+=res; //Vy(-l1,-l2,-l3) 
			}}
	}// end of for(il)
	}//end of the block, all other variable are inside this block expect mop
	/*
	 * notice that if we off reduntant component, we should change normalization coefficient 
	 */
	mop[0]/=(2.0*sqrt(2.0));
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[0], id + (*pos) * DEVPARAMS::Volume, gfoffset1);
	(*pos)++;
	if((arg.symmetry==5||arg.symmetry==6)&(offset!=0)){
		mop[1]/=(2.0*sqrt(2.0));
		GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[1], id + ((*pos)+offset) * DEVPARAMS::Volume, gfoffset1);
		(*pos)++;
	}
}

}
