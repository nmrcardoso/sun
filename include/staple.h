
#ifndef STAPLE_H
#define STAPLE_H


#include <complex.h>
#include <matrixsun.h>
#include <constants.h>
#include <index.h>
#include <device_load_save.h>


namespace CULQCD{




template <bool UseTex, ArrayType atype, class Real, bool SII_action>
__device__ inline void Staple_SI_SII_NO(complex *array, int mu, msun &staple, int id, int mustride, int muvolume, int offset){
	msun wsp = msun::zero();
	msun wtp = msun::zero();
	int newidmu1 = Index_4D_Neig_NM(id, mu, 1);
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		bool tau = ((nu==3||mu==3)?(true):(false));
		msun link;	
		int nuvolume = nu * mustride;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  id + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, nu, 1) + muvolume, offset);	
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + nuvolume, offset);
		if( tau ) wtp += link;
		else wsp += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_NM(id, nu, -1);
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, newidnum1  + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 1, nu,  -1) + nuvolume, offset);
		if( tau ) wtp += link;
		else wsp += link;
	}
	//calculate 2x1 spatial rectangular loops
	msun wsr = msun::zero();	
	//calculate one temporal and two spatial
	msun wstr = msun::zero();
	//calculate one temporal and two spatial
	msun wttr = msun::zero();
	
	//2x1 rectangular staples
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		msun link;	
		int nuvolume = nu * mustride;
		msun tmp;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  id + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 1, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 2) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 1) + muvolume, offset);
		tmp = link;
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_NM(id, mu, -1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_NM(id, mu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, -1, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_NM(id, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_NM(id, mu, 1) + nuvolume, offset);
		tmp += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_NM(id, nu, -1);
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  newidnum1 + muvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 1, nu, -1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 2, nu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + muvolume, offset);
		tmp += link;		
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_NM(id, mu, -1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, -1, nu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, -1, nu, -1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  newidnum1 + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 1, nu, -1) + nuvolume, offset);
		tmp += link;
		if( nu == 3 ) wstr += tmp;
		else if( mu == 3 ) wttr += tmp;
		else wsr += tmp;
	}
	
	
	//1x2 rectangular staples
	//if(mu!=3) 
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		msun link;	
		int nuvolume = nu * mustride;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  id + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, nu, 1) + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, nu, 2) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 1, nu, 1) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + nuvolume, offset);
		if( nu == 3 ) wttr += link;
		else if( mu == 3 ) wstr += link;
		else wsr += link;
		//wsr += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_NM(id, nu, -1);
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_NM(id, nu, -2) + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_NM(id, nu, -2) + muvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 1, nu, -2)  + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(id, mu, 1, nu, -1)  + nuvolume, offset);
		if( nu == 3 ) wttr += link;
		else if( mu == 3 ) wstr += link;
		else wsr += link;
		//wsr += link;
	}

	Real us2 = DEVPARAMS::IMP_Us * DEVPARAMS::IMP_Us;
	Real ut2 = DEVPARAMS::IMP_Ut * DEVPARAMS::IMP_Ut;
	Real us4 = us2 * us2;
	Real c0 = 5./(3. * us4 * DEVPARAMS::Aniso);
	Real c3 = DEVPARAMS::Aniso / (12. * us4 * ut2);
	if(SII_action){
		Real c1 = DEVPARAMS::Aniso * 4. / (3. * us2 * ut2);
		Real c2 = 1. / (12. * us2 * us4 * DEVPARAMS::Aniso);
		staple = wsp * c0 + wtp * c1 - wsr * c2 - wstr * c3;
	}
	else{
		Real c1 = DEVPARAMS::Aniso * 5. / (3. * us2 * ut2);
		Real c2 = 1. / (12. * us2 * us4 * DEVPARAMS::Aniso);
		Real c4 = DEVPARAMS::Aniso / (12. * us2 * ut2 * ut2);
		staple = wsp * c0 + wtp * c1 - wsr * c2 - wstr * c3 - wttr * c4;
	}
}



template <bool UseTex, ArrayType atype, class Real> 
__device__ void inline 
CalcStaple_NO(
	complex *array, 
	msun &staple, 
	int idx, 
	int mu
){
  int x[4];
  Index_4D_NM(idx, x);
  int mustride = DEVPARAMS::Volume;
  int muvolume = mu * mustride;
  int offset = DEVPARAMS::size;
	msun wsp = msun::zero();
	msun wtp = msun::zero();
	//int newidmu1 = Index_4D_Neig_NM(idx, mu, 1);
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		bool tau = ((nu==3||mu==3)?(true):(false));
    	int dx[4] = {0, 0, 0, 0};
		int nuvolume = nu * mustride;
		msun link;	
		//UP
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idx + nuvolume, offset);
		dx[nu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(x,dx) + muvolume, offset);	
		dx[nu]--;
		dx[mu]++;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_NM(x,dx) + nuvolume, offset);
		if( tau ) wtp += link;
		else wsp += link;
		dx[mu]--;
    	//DOWN
		dx[nu]--;
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_NM(x,dx) + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(x,dx)  + muvolume, offset);
		dx[mu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(x,dx) + nuvolume, offset);
		if( tau ) wtp += link;
		else wsp += link;
	}
	staple = wsp / DEVPARAMS::Aniso + wtp * DEVPARAMS::Aniso;
}



























__device__ __host__ inline int Index_4D_Neig_EO_1(const int x[], const int dx[], const int X[4]) {
	int y[4];
	for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
	int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
	
	int oddbit = (y[0] + y[1] + y[2] +y[3]) & 1;
	idx += oddbit  * param_HalfVolumeG();
	return idx;
}




template <bool UseTex, ArrayType atype, class Real>
__device__ inline void Staple_SOA12(complex *array, int mu, msun &staple, int x[4], int id, int oddbit, int idxoddbit, int mustride, int muvolume, int offset){
	msun wsp = msun::zero();
	msun wtp = msun::zero();
	for(int nu = 0; nu < 4; nu++) {
		if(mu != nu) {
			bool tau = ((nu==3||mu==3)?(true):(false));
		  	//int dx[4] = {0, 0, 0, 0};
		  	int dx[4];
		  	for(int i = 0; i < 4; ++i) dx[i] = 0;
			msun link;	
			int nuvolume = nu * mustride;
			link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
			dx[nu]++;
			link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO_1(x,dx,DEVPARAMS::GridWGhost) + muvolume, offset);	
			dx[nu]--;
			dx[mu]++;
			link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO_1(x,dx,DEVPARAMS::GridWGhost) + nuvolume, offset);
			if( tau ) wtp += link;
			else wsp += link;

			dx[mu]--;
			dx[nu]--;
			link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO_1(x,dx,DEVPARAMS::GridWGhost) + nuvolume, offset);	
			link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO_1(x,dx,DEVPARAMS::GridWGhost)  + muvolume, offset);
			dx[mu]++;
			link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO_1(x,dx,DEVPARAMS::GridWGhost) + nuvolume, offset);		
			if( tau ) wtp += link;
			else wsp += link;
		}
	}
	staple = wsp / DEVPARAMS::Aniso + wtp * DEVPARAMS::Aniso;
}












template <bool UseTex, ArrayType atype, class Real>
__device__ inline void Staple(complex *array, int mu, msun &staple, int id, int oddbit, int idxoddbit, int mustride, int muvolume, int offset){
	msun wsp = msun::zero();
	msun wtp = msun::zero();
	int newidmu1 = Index_4D_Neig_EO(id, oddbit, mu, 1);
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		bool tau = ((nu==3||mu==3)?(true):(false));
		msun link;	
		int nuvolume = nu * mustride;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);	
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + nuvolume, offset);
		if( tau ) wtp += link;
		else wsp += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_EO(id, oddbit, nu, -1);
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, newidnum1  + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu,  -1) + nuvolume, offset);
		if( tau ) wtp += link;
		else wsp += link;
	}
	staple = wsp / DEVPARAMS::Aniso + wtp * DEVPARAMS::Aniso;
/*	int newidmu1 = Index_4D_Neig_EO(id, oddbit, mu, 1);
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
	*/
}


template <bool UseTex, ArrayType atype, class Real, bool SII_action>
__device__ inline void Staple_SI_SII_(complex *array, int mu, msun &staple, int id, int oddbit, int idxoddbit, int mustride, int muvolume, int offset){
	msun wsp = msun::zero();
	msun wtp = msun::zero();
	//calculate 2x1 spatial rectangular loops
	msun wsr = msun::zero();	
	//calculate one temporal and two spatial
	msun wstr = msun::zero();
	//calculate one temporal and two spatial
	msun wttr = msun::zero();
	
	int newidmu1 = Index_4D_Neig_EO(id, oddbit, mu, 1);
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		bool tau = ((nu==3||mu==3)?(true):(false));
		msun link_1x1, link2x1_a, link2x1_b, link1x2_a, link1x2_b;	
		int nuvolume = nu * mustride;
		//UP	
		//1x1
		link1x2_a = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);		 
		link2x1_b = GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);
		link2x1_a = link1x2_a * link2x1_b;
		link1x2_b = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + nuvolume, offset);
		link_1x1 = link2x1_a * link1x2_b;
		link2x1_b *= link1x2_b;
		if( tau ) wtp += link_1x1;
		else wsp += link_1x1;
		//2x1 rectangular staples		
		link2x1_a *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, 1) + muvolume, offset);
		link2x1_a *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 2) + nuvolume, offset);
		link2x1_a *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1) + muvolume, offset);
		
		link_1x1 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, -1) + muvolume, offset);
		link_1x1 *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, -1) + nuvolume, offset);
		link_1x1 *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, -1, nu, 1) + muvolume, offset);
		link2x1_a += link_1x1 * link2x1_b;
		if( nu == 3 ) wstr += link2x1_a;
		else if( mu == 3 ) wttr += link2x1_a;
		else wsr += link2x1_a;
		//1x2 rectangular staples;	
		link1x2_a *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + nuvolume, offset);
		link1x2_a *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 2) + muvolume, offset);
		link1x2_a *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, 1) + nuvolume, offset);
		link1x2_a *= link1x2_b;
		if( nu == 3 ) wttr += link1x2_a;
		else if( mu == 3 ) wstr += link1x2_a;
		else wsr += link1x2_a;
		
		
		//DOWN
		//1x1	
		int newidnum1 = Index_4D_Neig_EO(id, oddbit, nu, -1);		
		link1x2_a = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);		 
		link2x1_b = GAUGE_LOAD<UseTex, atype, Real>( array, newidnum1  + muvolume, offset);
		link2x1_a = link1x2_a * link2x1_b;
		link1x2_b = GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu,  -1) + nuvolume, offset);
		link_1x1 = link2x1_a * link1x2_b;
		link2x1_b *= link1x2_b;
		if( tau ) wtp += link_1x1;
		else wsp += link_1x1;
		
		//2x1 rectangular staples		
		link2x1_a *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -1) + muvolume, offset);
		link2x1_a *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 2, nu, -1) + nuvolume, offset);
		link2x1_a *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + muvolume, offset);
		
		link_1x1 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, -1) + muvolume, offset);
		link_1x1 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, -1, nu, -1) + nuvolume, offset);
		link_1x1 *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, -1, nu, -1) + muvolume, offset);
		link2x1_a += link_1x1 * link2x1_b;
		if( nu == 3 ) wstr += link2x1_a;
		else if( mu == 3 ) wttr += link2x1_a;
		else wsr += link2x1_a;
		
		//1x2 rectangular staples;
		link1x2_a *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, nu, -2) + nuvolume, offset);	
		link1x2_a *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, nu, -2) + muvolume, offset);	
		link1x2_a *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -2)  + nuvolume, offset);
		link1x2_a *= link1x2_b;
		if( nu == 3 ) wttr += link1x2_a;
		else if( mu == 3 ) wstr += link1x2_a;
		else wsr += link1x2_a;
		
	}

	Real us2 = DEVPARAMS::IMP_Us * DEVPARAMS::IMP_Us;
	Real ut2 = DEVPARAMS::IMP_Ut * DEVPARAMS::IMP_Ut;
	Real us4 = us2 * us2;
	Real c0 = 5./(3. * us4 * DEVPARAMS::Aniso);
	Real c3 = DEVPARAMS::Aniso / (12. * us4 * ut2);
	if(SII_action){
		Real c1 = DEVPARAMS::Aniso * 4. / (3. * us2 * ut2);
		Real c2 = 1. / (12. * us2 * us4 * DEVPARAMS::Aniso);
		staple = wsp * c0 + wtp * c1 - wsr * c2 - wstr * c3;
	}
	else{
		Real c1 = DEVPARAMS::Aniso * 5. / (3. * us2 * ut2);
		Real c2 = 1. / (12. * us2 * us4 * DEVPARAMS::Aniso);
		Real c4 = DEVPARAMS::Aniso / (12. * us2 * ut2 * ut2);
		staple = wsp * c0 + wtp * c1 - wsr * c2 - wstr * c3 - wttr * c4;
	}
}






template <bool UseTex, ArrayType atype, class Real, bool SII_action>
__device__ inline void Staple_SI_SII(complex *array, int mu, msun &staple, int id, int oddbit, int idxoddbit, int mustride, int muvolume, int offset){
	msun wsp = msun::zero();
	msun wtp = msun::zero();
	int newidmu1 = Index_4D_Neig_EO(id, oddbit, mu, 1);
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		bool tau = ((nu==3||mu==3)?(true):(false));
		msun link;	
		int nuvolume = nu * mustride;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);	
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + nuvolume, offset);
		if( tau ) wtp += link;
		else wsp += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_EO(id, oddbit, nu, -1);
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, newidnum1  + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu,  -1) + nuvolume, offset);
		if( tau ) wtp += link;
		else wsp += link;
	}
	//calculate 2x1 spatial rectangular loops
	msun wsr = msun::zero();	
	//calculate one temporal and two spatial
	msun wstr = msun::zero();
	//calculate one temporal and two spatial
	msun wttr = msun::zero();
	
	//2x1 rectangular staples
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		msun link;	
		int nuvolume = nu * mustride;
		msun tmp;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 2) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1) + muvolume, offset);
		tmp = link;
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, -1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, -1, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, 1) + nuvolume, offset);
		tmp += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_EO(id, oddbit, nu, -1);
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  newidnum1 + muvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 2, nu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + muvolume, offset);
		tmp += link;		
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, -1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, -1, nu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, -1, nu, -1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  newidnum1 + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -1) + nuvolume, offset);
		tmp += link;
		if( nu == 3 ) wstr += tmp;
		else if( mu == 3 ) wttr += tmp;
		else wsr += tmp;
	}
	
	
	//1x2 rectangular staples
	//if(mu!=3) 
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		msun link;	
		int nuvolume = nu * mustride;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 2) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, 1) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidmu1 + nuvolume, offset);
		if( nu == 3 ) wttr += link;
		else if( mu == 3 ) wstr += link;
		else wsr += link;
		//wsr += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_EO(id, oddbit, nu, -1);
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, nu, -2) + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, nu, -2) + muvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -2)  + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -1)  + nuvolume, offset);
		if( nu == 3 ) wttr += link;
		else if( mu == 3 ) wstr += link;
		else wsr += link;
		//wsr += link;
	}

	Real us2 = DEVPARAMS::IMP_Us * DEVPARAMS::IMP_Us;
	Real ut2 = DEVPARAMS::IMP_Ut * DEVPARAMS::IMP_Ut;
	Real us4 = us2 * us2;
	Real c0 = 5./(3. * us4 * DEVPARAMS::Aniso);
	Real c3 = DEVPARAMS::Aniso / (12. * us4 * ut2);
	if(SII_action){
		Real c1 = DEVPARAMS::Aniso * 4. / (3. * us2 * ut2);
		Real c2 = 1. / (12. * us2 * us4 * DEVPARAMS::Aniso);
		staple = wsp * c0 + wtp * c1 - wsr * c2 - wstr * c3;
	}
	else{
		Real c1 = DEVPARAMS::Aniso * 5. / (3. * us2 * ut2);
		Real c2 = 1. / (12. * us2 * us4 * DEVPARAMS::Aniso);
		Real c4 = DEVPARAMS::Aniso / (12. * us2 * ut2 * ut2);
		staple = wsp * c0 + wtp * c1 - wsr * c2 - wstr * c3 - wttr * c4;
	}
}





template <bool UseTex, ArrayType atype, class Real, bool SII_action>
__device__ inline void Staple_SI_SII_a(complex *array, int mu, msun &staple, int id, int oddbit, int idxoddbit, int mustride, int muvolume, int offset){
	msun wsp = msun::zero();
	msun wtp = msun::zero();
	int newidmu1 = Index_4D_Neig_EO(id, oddbit, mu, 1);
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		bool tau = ((nu==3||mu==3)?(true):(false));
		msun link;	
		int nuvolume = nu * mustride;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array, newidmu1 + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);	
		if( tau ) wtp += link;
		else wsp += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_EO(id, oddbit, nu, -1);		
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu,  -1) + nuvolume, offset);	
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, newidnum1  + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);
		if( tau ) wtp += link;
		else wsp += link;
	}
	//calculate 2x1 spatial rectangular loops
	msun wsr = msun::zero();	
	//calculate one temporal and two spatial
	msun wstr = msun::zero();
	//calculate one temporal and two spatial
	msun wttr = msun::zero();
	
	//2x1 rectangular staples
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		msun link;	
		int nuvolume = nu * mustride;
		msun tmp;
		//UP	
		// positive foward
		link = GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 2) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		tmp = link;
		// positive backward
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, 1) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, -1, nu, 1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, -1) + muvolume, offset);
		tmp += link;
		
		
		
		//DOWN	
		int newidnum1 = Index_4D_Neig_EO(id, oddbit, nu, -1);
		// negative foward
		link = GAUGE_LOAD<UseTex, atype, Real>( array, newidmu1 + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 2, nu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -1) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + muvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);
		tmp += link;		
		// negative backward
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  newidnum1 + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, -1, nu, -1) + muvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, -1, nu, -1) + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, mu, -1) + muvolume, offset);
		tmp += link;
		if( nu == 3 ) wstr += tmp;
		else if( mu == 3 ) wttr += tmp;
		else wsr += tmp;
	}
	
	
	//1x2 rectangular staples
	//if(mu!=3) 
	for(int nu = 0; nu < 4; nu++){ if(mu == nu) continue;
		msun link;	
		int nuvolume = nu * mustride;
		//UP	
		link = GAUGE_LOAD<UseTex, atype, Real>( array, newidmu1 + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, 1) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 2) + muvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, nu, 1) + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  idxoddbit + nuvolume, offset);
		if( nu == 3 ) wttr += link;
		else if( mu == 3 ) wstr += link;
		else wsr += link;
		//wsr += link;
		//DOWN	
		int newidnum1 = Index_4D_Neig_EO(id, oddbit, nu, -1);	
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -1)  + nuvolume, offset);
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(id, oddbit, mu, 1, nu, -2)  + nuvolume, offset);	
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, nu, -2) + muvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_EO(id, oddbit, nu, -2) + nuvolume, offset);
		link *= GAUGE_LOAD<UseTex, atype, Real>( array,  newidnum1 + nuvolume, offset);
		if( nu == 3 ) wttr += link;
		else if( mu == 3 ) wstr += link;
		else wsr += link;
		//wsr += link;
	}

	Real us2 = DEVPARAMS::IMP_Us * DEVPARAMS::IMP_Us;
	Real ut2 = DEVPARAMS::IMP_Ut * DEVPARAMS::IMP_Ut;
	Real us4 = us2 * us2;
	Real c0 = 5./(3. * us4 * DEVPARAMS::Aniso);
	Real c3 = DEVPARAMS::Aniso / (12. * us4 * ut2);
	if(SII_action){
		Real c1 = DEVPARAMS::Aniso * 4. / (3. * us2 * ut2);
		Real c2 = 1. / (12. * us2 * us4 * DEVPARAMS::Aniso);
		staple = wsp * c0 + wtp * c1 - wsr * c2 - wstr * c3;
	}
	else{
		Real c1 = DEVPARAMS::Aniso * 5. / (3. * us2 * ut2);
		Real c2 = 1. / (12. * us2 * us4 * DEVPARAMS::Aniso);
		Real c4 = DEVPARAMS::Aniso / (12. * us2 * ut2 * ut2);
		staple = wsp * c0 + wtp * c1 - wsr * c2 - wstr * c3 - wttr * c4;
	}
}





/**
  @brief Calculate the staple along direction mu and nu in even odd lattice array
  @param array gauge field
  @param staple store staple result
  @param idx 1D lattice index
  @param mu direction
  @param nu direction
  @param oddbit parity of the current lattice site.
*/
template <bool UseTex, ArrayType atype, class Real> 
__device__ void inline Staple(
        complex *array, 
        msun &staple, 
        int idx, 
        int mu, 
        int nu,
        int oddbit
        ){
	msun link;	
	int nuvolume = nu * DEVPARAMS::Volume;
	int muvolume = mu * DEVPARAMS::Volume;
    //UP	
    link = GAUGE_LOAD<UseTex, atype, Real>( array,  idx + oddbit * DEVPARAMS::HalfVolume + nuvolume);
    link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(idx, oddbit, nu, 1) +muvolume );	
    link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_EO(idx, oddbit, mu, 1) + nuvolume );
    staple += link;
    //DOWN	
    link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_4D_Neig_EO(idx, oddbit, nu, -1) +nuvolume );	
    link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(idx, oddbit, nu, -1)  + muvolume);
    link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_EO(idx, oddbit, mu, 1, nu,  -1)+ nuvolume);
    staple += link;
}

/**
  @brief Calculate the staple along direction mu and nu in normal lattice array.
  @param array gauge field
  @param staple store staple result
  @param idx 1D lattice index
  @param mu direction
  @param nu direction
*/
template <bool UseTex, ArrayType atype, class Real> 
__device__ void inline Staple(
        complex *array, 
        msun &staple, 
        int idx, 
        int mu, 
        int nu
        ){
	msun link;	
	int nuvolume = nu * DEVPARAMS::Volume;
	int muvolume = mu * DEVPARAMS::Volume;
    //UP	
    link = GAUGE_LOAD<UseTex, atype, Real>( array, idx + nuvolume);
    link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(idx, nu, 1) + muvolume);
    link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_NM(idx, mu, 1) + nuvolume);
    staple += link;
    //DOWN	
    link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_4D_Neig_NM(idx, nu, -1) + nuvolume);
    link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(idx, nu, -1) + muvolume);
    link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_4D_Neig_NM(idx, mu, 1, nu, -1) + nuvolume);
    staple += link;
}

/*template <bool evenoddarray, bool UseTex, ArrayType atype, class Real> 
  __device__ void inline 
  Staple(
  complex *array, 
  msun &staple, 
  int idx, 
  int mu, 
  int nu,
  int oddbit
  ){
  msun link;	
  int nuvolume = nu * DEVPARAMS::Volume;
  int muvolume = mu * DEVPARAMS::Volume;
  if(evenoddarray){
  //UP	
  link = GAUGE_LOAD<UseTex, atype, Real>( array,  idx + oddbit * DEVPARAMS::HalfVolume + nuvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborEOIndex(idx, oddbit, nu, 1) +muvolume );	
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborEOIndex(idx, oddbit, mu, 1) + nuvolume ).dagger();
  staple += link;
  //DOWN	
  link = GAUGE_LOAD<UseTex, atype, Real>( array,  neighborEOIndex(idx, oddbit, nu, -1) +nuvolume ).dagger();	
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborEOIndex(idx, oddbit, nu, -1)  + muvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborEOIndex(idx, oddbit, mu, 1, nu,  -1)+ nuvolume);
  staple += link;
  }
  else{
  //UP	
  link = GAUGE_LOAD<UseTex, atype, Real>( array, idx + nuvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, nu, 1) + muvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, mu, 1) + nuvolume).dagger();
  staple += link;
  //DOWN	
  link = GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, nu, -1) + nuvolume).dagger();
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, nu, -1) + muvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, mu, 1, nu, -1) + nuvolume);
  staple += link;
  }

  }*/

}
#endif

