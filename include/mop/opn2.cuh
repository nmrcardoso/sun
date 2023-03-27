#pragma once
#include<mop/wloop_arg.h>
namespace CULQCD{
//Nuno operators with 
//////// 2nd nuno operator /////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE void N2( WLOPArg<Real> arg,int id, int l1, int l2, int muvolume, msun line_left, msun line_right, int gfoffset1, int *pos, int offset){
//    if(id==0)printf("the symmetry inside N2 is %d\n",arg.symmetry);
	msun mop[2]={msun::zero(),msun::zero()};
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
    //   ____                                        //
	//  /   /                                        //
	//  |   |____ line                               //
	//  svec[i]                                      //
	///////////////////////////////////////////////////
	{
	int dir1 = arg.mu;
	int dir2 = (dir1+1)%3;
	int dir3 = (dir1+2)%3;
	int dmu[2]; int dmu1[2];
	dmu[0] = dir2;
	dmu[1] = dir3;
	dmu1[0] = dir3;
	dmu1[1] = dir2;
	{
	int halfline = arg.radius/2;
	//2 comp, 1st upway 2nd foward  0/1
	for(int il = 0; il < 2; il++){
		int ids=id;
		int ids_r=Index_4D_Neig_NM(id, dir1, halfline);
		msun left=msun::identity();
		msun right=msun::identity();
		for(int i1=0;i1<l1;i1++){
		left *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_4D_Neig_NM(ids, dmu[il],1);
		right *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids_r + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		ids_r=Index_4D_Neig_NM(ids_r, dmu[il],1);
		}
		right=right.dagger();//ok
		//2nd forward 
		msun link0 =msun::identity();
		for(int i2=0;i2<l2;i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu1[il],1);
		}
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0; i2<l2;i2++){
			ids= Index_4D_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		////////////////////////////////////////
		//////////// adding item ///////////////
		////////////////////////////////////////
		
		{
		msun res=left*link0*right*line_right;
		if(arg.symmetry==0 ||arg.symmetry==2)mop[0] += res; //sigma_g_plus, sigma_u_plus
		if(arg.symmetry==1 || arg.symmetry==3){//sigma_g_minus, sigma_u_minus
			if(il==0)mop[0]+=res;
			if(il==1)mop[0]-=res;
		}
		if(arg.symmetry==4 || arg.symmetry==5){//pi_g, pi_u
			if(il==0){
				mop[0]+=res;
				mop[1]+=res;
			}
			if(il==1) {
				mop[0]+=timesI(res);
				mop[1]-=timesI(res);
				}
			}
		if(arg.symmetry==6 ||arg.symmetry==7){//delta_g, delta_u
			if(il==0) {
				mop[0]+=res;
				mop[1]+=res;
			}
			if(il==1){
				mop[0]-=res;
				mop[1]+=res;
				}
		}
		}
		/////////////////////////////////////////
		////////////// 2nd backward /////////////
		/////////////////////////////////////////
		ids=Index_4D_Neig_NM(id, dmu[il], l1, dmu1[il], -l2);// pay attension to id and ids
		link0 =msun::identity(); //updating link0
		for(int i2=0; i2<l2;i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu1[il],1);
		}
		link0=link0.dagger();
		ids=Index_4D_Neig_NM(id, dmu[il], l1, dmu1[il], -l2);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0;i2<l2;i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu1[il],1);
		}
		/////////////////////////////////////////
		//////////// adding item ////////////////
		/////////////////////////////////////////
		{
		msun res= left*link0*right*line_right;
		if(arg.symmetry==0 || arg.symmetry==2)mop[0] += res;//sigma_g_plus, sigma_u_plus
		if(arg.symmetry==1 || arg.symmetry==3){//sigma_g_minus, sigma_u_minus
			if(il==1)mop[0]+=res;
			if(il==0)mop[0]-=res;
		}
		
		if(arg.symmetry==4 || arg.symmetry==5){//pi_g, pi_u
			if(il==1) {
				mop[0]+=timesI(res);
				mop[1]+=timesI(res);
			}
			if(il==0){
				mop[0]+=res;
				mop[1]-=res;
			}
		}
		if(arg.symmetry==6 ||arg.symmetry==7){//delta_g, delta_u
			if(il==1){
				mop[0]-=res;
				mop[1]-=res;
				}
			if(il==0){
				mop[0]+=res;
				mop[1]-=res;
			}}
		}
	} //end of loop direction
	//1st down, 2nd up
	for(int il = 0; il < 2; il++){
		int ids=Index_4D_Neig_NM(id, dmu[il], -l1);
		int ids_r=Index_4D_Neig_NM(id, dir1, halfline, dmu[il], -l1);
		msun left=msun::identity();
		msun right=msun::identity();
		for(int i1=0; i1<l1;i1++){
			left *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu[il], 1);
			right *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids_r + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids_r=Index_4D_Neig_NM(ids_r, dmu[il], 1);
		}
		left=left.dagger();
		ids=Index_4D_Neig_NM(id, dmu[il], -l1);
		///////// 2nd forward ////////////
		msun link0=msun::identity();
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu1[il],1);
		}
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0;i2<l2;i2++){
			ids= Index_4D_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		{
		msun res=left*link0*right*line_right;
		if(arg.symmetry==0 || arg.symmetry==2) mop[0] += res;//sigma_g_plus, sigma_u_plus
		if(arg.symmetry==1 || arg.symmetry==3){//sigma_g_minus, sigma_u_minus
			if(il==1)mop[0]+=res;
			if(il==0)mop[0]-=res;
		}
		if(arg.symmetry==4 || arg.symmetry==5){//pi_g, pi_u
			if(il==1){
				mop[0]-=timesI(res);
				mop[1]-=timesI(res);
				}
			if(il==0){
				mop[0]-=res;
				mop[1]+=res;
			}
		}
		if(arg.symmetry==6 ||arg.symmetry==7){//delta_g, delta_u
			if(il==1){
				mop[0]-=res;
				mop[1]-=res;
				}
			if(il==0){
				mop[0]+=res;
				mop[1]-=res;
				}
				}
		}
		/////////// 2nd backward /////////////////
		// updating coordinates
		ids=Index_4D_Neig_NM(id, dmu[il], -l1, dmu1[il], -l2);
		link0=msun::identity();
		for(int i2=0; i2<l2;i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dmu1[il], 1);
		}
		link0=link0.dagger();
		ids=Index_4D_Neig_NM(id, dmu[il], -l1, dmu1[il], -l2);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0;i2<l2;i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dmu1[il], 1);
		}
		{
			msun res=left*link0*right*line_right;
			if(arg.symmetry==0 || arg.symmetry==2) mop[0] += res;
			if(arg.symmetry==1 || arg.symmetry==3){//sigma_g_minus, sigma_u_minus
				if(il==0)mop[0]+=res;
				if(il==1)mop[0]-=res;
			}
			if(arg.symmetry==4 || arg.symmetry==5){//pi_g, pi_u
				if(il==0){
					mop[0]-=res;
					mop[1]-=res;
					}
				if(il==1){
					mop[0]-=timesI(res);
					mop[1]+=timesI(res);
				}
			}
			if(arg.symmetry==6 ||arg.symmetry==7){//delta_g, delta_u
				if(il==0){
					mop[0]+=res;
					mop[1]+=res;
				 }
				if(il==1){
					mop[0]-=res;
					mop[1]+=res;
					}
				}
			}
		}//end of loop for direction
		}//end of block for left staple


	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//       ____                                    //
	//      /   /                                    //
	//  ____|   | svec[i]                            //
	//   line                                        //
	///////////////////////////////////////////////////
	{
	int halfline = (arg.radius + 1) / 2;
	int s2 = Index_4D_Neig_NM( id, dir1, halfline);
    //2 comp, 1st upway 2nd foward 8/9
	for(int il = 0; il < 2; il++){
		int ids=s2;
		int ids_r=Index_4D_Neig_NM(id, dir1, arg.radius);
		msun left=msun::identity();
		msun right=msun::identity();
		for(int i1=0; i1<l1; i1++){
			left*= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids+ dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dmu[il], 1);
			right*= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids_r+ dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids_r = Index_4D_Neig_NM(ids_r, dmu[il], 1);
		}
		right=right.dagger();
		msun link0=msun::identity();
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dmu1[il], 1);
			}
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0; i2<l2; i2++){
			ids = Index_4D_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		{
		msun res=line_left*left*link0*right;//1st up, 2nd up
		if(arg.symmetry==0) mop[0] += res;
		if(arg.symmetry==2) mop[0] -= res;//sigma_u_plus
		if(arg.symmetry==1){//sigma_g_minus
		 if(il==0) mop[0] +=res;
		 if(il==1) mop[0] -=res;
		}
		// left side is similiar to sigma_g_minus
		if(arg.symmetry==3){//sigma_u_minus
		 if(il==0) mop[0] -=res;
		 if(il==1) mop[0] +=res;
		 }
		if(arg.symmetry==5){//pi_u
			if(il==0){
				mop[0]+=res;
				mop[1]+=res;
			}
			if(il==1){
				mop[0]+=timesI(res);
				mop[1]-=timesI(res);
			}
		}
		if(arg.symmetry==4){//pi_g
			if(il==0){
				mop[0]-=res;
				mop[1]-=res;
			}
			if(il==1){
				mop[0]-=timesI(res);
				mop[1]+=timesI(res);
			}
		}
		if(arg.symmetry==6){//delta_g
			if(il==0){
				mop[0]+=res;
				mop[1]+=res;
			}
			if(il==1){
				mop[0]-=res;
				mop[1]+=res;
			}
		}
		if(arg.symmetry==7){//delta_u
			if(il==0){
				mop[0]-=res;
				mop[1]-=res;
			}
			if(il==1){
				mop[0]+=res;
				mop[1]-=res;
			}
		}}
		//////// 2nd backward //////////
		//1st up, update coordinates
		ids=Index_4D_Neig_NM(s2, dmu[il], l1, dmu1[il],-l2);
		link0=msun::identity();
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu1[il], 1);
		}
		link0=link0.dagger();//2nd down
		ids=Index_4D_Neig_NM( s2, dmu[il], l1, dmu1[il],-l2);//1st up, 2nd down
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu1[il],1);
		}
		{
			msun res=line_left*left*link0*right;//1up,2nd down
			if(arg.symmetry==0) mop[0] += res;
			if(arg.symmetry==2) mop[0] -= res;//sigma_u_plus
			if(arg.symmetry==1){//sigma_g_minus
				if(il==1) mop[0]+=res;
				if(il==0) mop[0]-=res;
			}
			if(arg.symmetry==3){//sigma_u_minus
				if(il==1) mop[0]-=res;
				if(il==0) mop[0]+=res;
			}
			if(arg.symmetry==5){//pi_u
				if(il==1){
					mop[0]+=timesI(res);
					mop[1]+=timesI(res);
				}
				if(il==0){
					mop[0]+=res;
					mop[1]-=res;
				}}
			if(arg.symmetry==4){//pi_g
				if(il==1){
					mop[0]-=timesI(res);
					mop[1]-=timesI(res);
				}
				if(il==0){
					mop[0]-=res;
					mop[1]+=res;
				}
			}
			if(arg.symmetry==6){//delta_g
				if(il==1){
					mop[0]-=res;
					mop[1]-=res;
				}
				if(il==0){
					mop[0]+=res;
					mop[1]-=res;
				}
			}
			if(arg.symmetry==7){//delta_u
				if(il==1){
					mop[0]+=res;
					mop[1]+=res;
				}
				if(il==0){
					mop[0]-=res;
					mop[1]+=res;
				}
			}}
		}//end of loop for direction
	//2 comp, 1st downway 2nd foward 12/13
	for(int il = 0; il < 2; il++){
		int ids=Index_4D_Neig_NM(s2 , dmu[il], -l1);
		int ids_r=Index_4D_Neig_NM(id, dir1, arg.radius, dmu[il], -l1);
		msun left=msun::identity();
		msun right=msun::identity();
		for(int i1=0; i1<l1; i1++){
			left *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids+ dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_4D_Neig_NM(ids, dmu[il],1);
			right *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids_r+ dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids_r=Index_4D_Neig_NM(ids_r, dmu[il],1);//notice to definition
		}
		left=left.dagger();//1st down
		ids=Index_4D_Neig_NM(s2 , dmu[il], -l1);
		msun link0=msun::identity();
		// 2nd forward
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dmu1[il], 1);
		}
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0; i2<l2; i2++){
			ids= Index_4D_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		{
			msun res=line_left*left*link0*right;// notice right side already read as U
			if(arg.symmetry==0) mop[0] +=res;
			if(arg.symmetry==2) mop[0] -=res;//sigma_u_plus
			if(arg.symmetry==1){//sigma_g_minus
				if(il==1) mop[0]+=res;
				if(il==0) mop[0]-=res;
			}
			if(arg.symmetry==3){//sigma_u_minus
				if(il==1) mop[0]-=res;
				if(il==0) mop[0]+=res;
			}
			if(arg.symmetry==5){//pi_u
				if(il==1){
					mop[0]-=timesI(res);
					mop[1]-=timesI(res);
				}
				if(il==0){
					mop[0]-=res;
					mop[1]+=res;
				}
			}
			if(arg.symmetry==4){//pi_g
				if(il==1){
					mop[0]+=timesI(res);
					mop[1]+=timesI(res);
					}
				if(il==0){
					mop[0]+=res;
					mop[1]-=res;
				}
			}
			if(arg.symmetry==6){//delta_g
				if(il==1){
					mop[0]-=res;
					mop[1]-=res;
				}
				if(il==0){
					mop[0]+=res;
					mop[1]-=res;
				}
			}
			if(arg.symmetry==7){//delta_u
				if(il==1){
					mop[0]+=res;
					mop[1]+=res;
					}
				if(il==0){
					mop[0]-=res;
					mop[1]+=res;
				}
			}}
	//2 comp, 1st downway 2nd backward 6/7
		//update items
		ids=Index_4D_Neig_NM(s2, dmu[il], -l1, dmu1[il], -l2);
		link0=msun::identity();
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dmu1[il], 1);
		}
		link0=link0.dagger(); //2nd comp is back
		ids=Index_4D_Neig_NM(s2, dmu[il], -l1, dmu1[il], -l2);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids , dir1, 1);
		}
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_4D_Neig_NM(ids, dmu1[il], 1);
		}
		/////////// add item //////////////
		{
			msun res=line_left*left*link0*right;
			if(arg.symmetry==0) mop[0] += res;
			if(arg.symmetry==2) mop[0] -= res;//sigma_u_plus
			if(arg.symmetry==1){//sigma_g_minus
				if(il==0) mop[0]+=res;
				if(il==1) mop[0]-=res;
			}
			if(arg.symmetry==3){//sigma_u_minus
				if(il==0) mop[0]-=res;
				if(il==1) mop[0]+=res;
			}
			if(arg.symmetry==5){//pi_u
				if(il==0){
					mop[0]-=res;
					mop[1]-=res;
				}
				if(il==1){
					mop[0]-=timesI(res);
					mop[1]+=timesI(res);}
			}
			if(arg.symmetry==4){//pi_g
				if(il==0){
					mop[0]+=res;
					mop[1]+=res;
				}
				if(il==1){
					mop[0]+=timesI(res);
					mop[1]-=timesI(res);
				}
			}
			if(arg.symmetry==6){//delta_g
				if(il==0){
					mop[0]+=res;
					mop[1]+=res;
					}
				if(il==1){
					mop[0]-=res;
					mop[1]+=res;
					}
				}
			if(arg.symmetry==7){//delta_u
				if(il==0){
					mop[0]-=res;
					mop[1]-=res;
				}
				if(il==1){
					mop[0]+=res;
					mop[1]-=res;
				}
			}
			}
		}//end of dir loop
		}//end of block for right staple
	}// end of block containing the main part
	mop[0] *= 0.25;
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[0], id + (*pos) * DEVPARAMS::Volume, gfoffset1);
	(*pos)++;
	if((arg.symmetry==4 || arg.symmetry==5 ||arg.symmetry==6||arg.symmetry==7)& (offset!=0)){
		mop[1] *= 0.25;
		GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[1], id + ((*pos)+offset) * DEVPARAMS::Volume, gfoffset1);
		(*pos)++;
	}
}

}
