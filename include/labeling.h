#pragma once
#include<meas/wloopex.h>
#include<cstring>
#include <time.h>

namespace CULQCD{
string get_name_sym(int name){
	switch (name){
		case sigma_g_plus:
			return "sigma_gp";
			break;
		case sigma_g_minus:
			return "sigma_gm";
			break;
		case sigma_u_plus:
			return "sigma_up";
			break;
		case sigma_u_minus:
			return "sigma_um";
			break;
		case pi_g:
			return "pi_g";
			break;
		case pi_u:
			return "pi_u";
			break;
		case delta_g:
			return "delta_g";
		case delta_u:
			return "delta_u";
		default :
		    printf("symmetry channel is not included %d,%s",__LINE__,__FUNCTION__);
		    throw "symmetry channel is not included";
		    return 0;
		}
}}

