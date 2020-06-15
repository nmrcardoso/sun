
#ifndef HOSTDEVICE_INDEX_H
#define HOSTDEVICE_INDEX_H


#include <vector_types.h>
#include <constants.h>



namespace CULQCD{


////////////////////////////////////////////////////////////////////
///////////   Device Functions  ////////////////////////////////////
////////////////////////////////////////////////////////////////////



//From normal to normal lattice index
//4D
inline  __host__   __device__ void Index_4D_NM(const int id, int x[4]){
	x[3] = id/(param_Grid(0) * param_Grid(1) * param_Grid(2));
	x[2] = (id/(param_Grid(0) * param_Grid(1))) % param_Grid(2);
	x[1] = (id/param_Grid(0)) % param_Grid(1);
	x[0] = id % param_Grid(0); 
}


__device__ __host__ inline int Index_4D_NM(const int y[]) {
  return (((y[3]*param_Grid(2) + y[2])*param_Grid(1) + y[1])*param_Grid(0) + y[0]);
}



inline  __host__  __device__ int Index_4D_Neig_NM(const int id, const int mu, const int lmu, const int nu, const int lnu){
	int x[4];
	Index_4D_NM(id, x);
	x[mu] = (x[mu]+lmu+param_Grid(mu)) % param_Grid(mu);
	x[nu] = (x[nu]+lnu+param_Grid(nu)) % param_Grid(nu);
	return (((x[3] * param_Grid(2) + x[2]) * param_Grid(1)) + x[1] ) * param_Grid(0) + x[0];
}

inline  __host__   __device__ int Index_4D_Neig_NM(const int id, const int mu, const int lmu){
	int x[4];
	Index_4D_NM(id, x); 
	x[mu] = (x[mu]+lmu+param_Grid(mu)) % param_Grid(mu);
	return (((x[3] * param_Grid(2) + x[2]) * param_Grid(1)) + x[1] ) * param_Grid(0) + x[0];
}

__device__ __host__ inline int Index_4D_Neig_NM(const int x[], const int mu, const int lmu) {
	int y[4];
	for (int i=0; i<4; i++) y[i] = x[i];
    y[mu] = (x[mu]+lmu+param_Grid(mu)) % param_Grid(mu);
	return (((y[3]*param_Grid(2) + y[2])*param_Grid(1) + y[1])*param_Grid(0) + y[0]);
}


__device__ __host__ inline int Index_4D_Neig_NM(const int x[], int dx[]) {
	int y[4];
	for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + param_Grid(i)) % param_Grid(i);
	return (((y[3]*param_Grid(2) + y[2])*param_Grid(1) + y[1])*param_Grid(0) + y[0]);
}


inline  __host__   __device__ void Index_4D_Neig_NM(int y[], int x[], int mu, int lmu){
  for (int i=0; i<4; i++) y[i] = x[i];
  y[mu] = (y[mu] + lmu + param_Grid(mu)) % param_Grid(mu);
}
inline  __host__   __device__ void Index_4D_Neig_NM(int y[], int x[], int mu, int lmu, int nu, int lnu){
  for (int i=0; i<4; i++) y[i] = x[i];
  y[mu] = (y[mu] + lmu + param_Grid(mu)) % param_Grid(mu);
  y[nu] = (y[nu] + lnu + param_Grid(nu)) % param_Grid(nu);
}



//3D
inline  __host__   __device__ void Index_3D_NM(const int id, int x[3]){
  x[2] = id/(param_Grid(0) * param_Grid(1));
  x[1] = (id/param_Grid(0)) % param_Grid(1);
  x[0] = id % param_Grid(0); 
}
inline  __host__   __device__ int Index_3D_Neig_NM(const int id, int mu, int r){
	int x[3];
	Index_3D_NM(id, x);
	x[mu] = (x[mu] + r + param_Grid(mu)) % param_Grid(mu);
	return (x[2] * param_Grid(1) + x[1] ) * param_Grid(0) + x[0];
}
		
__device__ __host__ inline int Index_3D_Neig_NM(const int x[], const int dx[]) {
  int y[3];
  for (int i=0; i<3; i++) y[i] = (x[i] + dx[i] + param_Grid(i)) % param_Grid(i);
  return (((y[2])*param_Grid(1) + y[1])*param_Grid(0) + y[0]);
}

inline  __host__   __device__ int Index_3D_Neig_NM(const int x[3], int mu, int lmu, int nu, int lnu){
int y[3];
  for(int dir = 0; dir < 3; dir++) y[dir] = x[dir];
  y[mu] = (y[mu]+lmu + param_Grid(mu)) % param_Grid(mu);
  y[nu] = (y[nu]+lnu + param_Grid(nu)) % param_Grid(nu);
  return (((y[2]) * param_Grid(1)) + y[1] ) * param_Grid(0) + y[0];
} 

inline  __host__   __device__ int Index_3D_Neig_NM(const int x[3], int mu, int lmu){
int y[3];
  for(int dir = 0; dir < 3; dir++) y[dir] = x[dir];
  y[mu] = (y[mu]+lmu + param_Grid(mu)) % param_Grid(mu);
  return (((y[2]) * param_Grid(1)) + y[1] ) * param_Grid(0) + y[0];
} 



















//From EO to EO lattice index
//4D

inline  __host__   __device__ void Index_4D_EO(int x[4], const int id, const int oddbit){
	int za = (id / (param_Grid(0)/2));
	int zb =  (za / param_Grid(1));
	x[1] = za - zb * param_Grid(1);
	x[3] = (zb / param_Grid(2));
	x[2] = zb - x[3] * param_Grid(2);
	int xodd = (x[1] + x[2] + x[3] + oddbit) & 1;
	x[0] = (2 * id + xodd )  - za * param_Grid(0);
}


__device__ __host__ inline void Index_4D_EO(int x[4], const int id, const int parity, const int X[4]) {
	int za = (id / (X[0]/2));
	int zb =  (za / X[1]);
	x[1] = za - zb * X[1];
	x[3] = (zb / X[2]);
	x[2] = zb - x[3] * X[2];
	int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
	x[0] = (2 * id + x1odd)  - za * X[0];
	return;
}



__device__ __host__ inline int Index_4D_Neig_EO(const int x[], const int dx[], const int X[4]) {
	int y[4];
	for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
	int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
	return idx;
}





/**
	@brief U(id + lmu * e_mu), retrieves the neighbor in evenodd lattice index
*/
inline  __host__   __device__ int Index_4D_Neig_EO(const int id, int oddbit, int mu, int lmu){
	int x[4];
	Index_4D_EO(x, id, oddbit);
	#ifdef MULTI_GPU
	for(int i=0; i<4;i++) x[i] += param_border(i);
	#endif
	x[mu] = (x[mu]+lmu+param_GridG(mu)) % param_GridG(mu);
	
	int pos = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) / 2 ;
	int oddbit1 = (x[0] + x[1] + x[2] +x[3]) & 1;
	pos += oddbit1  * param_HalfVolumeG();
	return pos;
}


/**
	@brief U(id + lmu * e_mu + lnu * e_nu), retrieves the neighbor in evenodd lattice index
*/
inline  __host__   __device__ int Index_4D_Neig_EO(const int id, int oddbit, int mu, int lmu, int nu, int lnu){
	int x[4];
	Index_4D_EO(x, id, oddbit);
	#ifdef MULTI_GPU
	for(int i=0; i<4;i++) x[i] += param_border(i);
	#endif
	x[mu] = (x[mu]+lmu+param_GridG(mu)) % param_GridG(mu);
	x[nu] = (x[nu]+lnu+param_GridG(nu)) % param_GridG(nu);

	int pos = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) / 2 ;
	int oddbit1 = (x[0] + x[1] + x[2] +x[3]) & 1;
	pos += oddbit1  * param_HalfVolumeG();
	return pos;
}







/**
	@brief U(id + lmu * e_mu), retrieves the neighbor in evenodd lattice index
*/
inline  __host__   __device__ int Index_4D_Neig_EO(const int y[], int oddbit, int mu, int lmu){
	int x[4];
	#ifdef MULTI_GPU
	for(int i=0; i<4;i++) x[i] = y[i] + param_border(i);
	#else
	for(int i = 0; i<4; i++) x[i] = y[i];
	#endif
	x[mu] = (x[mu]+lmu+param_GridG(mu)) % param_GridG(mu);
	int pos = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) / 2 ;
	int oddbit1 = (x[0] + x[1] + x[2] +x[3]) & 1;
	pos += oddbit1  * param_HalfVolumeG();
	return pos;
}


/**
	@brief U(id + lmu * e_mu + lnu * e_nu), retrieves the neighbor in evenodd lattice index
*/
inline  __host__   __device__ int Index_4D_Neig_EO(const int y[], int oddbit, int mu, int lmu, int nu, int lnu){
	int x[4];
	#ifdef MULTI_GPU
	for(int i=0; i<4;i++) x[i] = y[i] + param_border(i);
	#else
	for(int i = 0; i<4; i++) x[i] = y[i];
	#endif
	x[mu] = (x[mu]+lmu+param_GridG(mu)) % param_GridG(mu);
	x[nu] = (x[nu]+lnu+param_GridG(nu)) % param_GridG(nu);
	int pos = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) / 2 ;
	int oddbit1 = (x[0] + x[1] + x[2] +x[3]) & 1;
	pos += oddbit1  * param_HalfVolumeG();
	return pos;
}












/**
	@brief U(id + e_mu), retrieves the neighbor in evenodd lattice index
*/
inline  __device__ int Index_4D_Neig_EO_PlusOne(const int id, const int oddbit, const int mu){
	int x[4];
	Index_4D_EO(x, id, oddbit);
	#ifdef MULTI_GPU
	for(int i=0; i<4;i++) x[i] += param_border(i);
	#endif
	x[mu] = (x[mu] + 1) % param_GridG(mu);
	int idx = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) >> 1;
	idx += (1 - oddbit)  * param_HalfVolumeG();
	return idx;
}


/**
	@brief U(id - e_mu), retrieves the neighbor in evenodd lattice index
*/
inline  __device__ int Index_4D_Neig_EO_MinusOne(const int id, const int oddbit, const int mu){
	int x[4];
	Index_4D_EO(x, id, oddbit);
	#ifdef MULTI_GPU
	for(int i=0; i<4;i++) x[i] += param_border(i);
	#endif
	x[mu] = (x[mu] - 1 + param_GridG(mu)) % param_GridG(mu);
	int idx = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) >> 1;
	idx += (1 - oddbit)  * param_HalfVolumeG();
	return idx;
}





//3D
























































#ifdef __CUDACC__
//#define __float2int_rd(a) a
#define	_mul(a, b)	((a) * (b))







/**
	@brief Returns a int3 the global 3D thread index for a 3D thread block.
*/
__forceinline__ int4 __device__ INDEX4D(){
  int4 id;
  int ij     = _mul(blockIdx.x,blockDim.x) + threadIdx.x;
  id.y = __float2int_rd(ij/param_Grid(0));
  id.x = ij - id.y * param_Grid(0);
  id.z     = _mul(blockIdx.y ,blockDim.y) + threadIdx.y;
  id.w     = _mul(blockIdx.z ,blockDim.z) + threadIdx.z;
  return id;
}
/**
	@brief Returns a int3 the global 3D thread index for a 3D thread block.
*/
__forceinline__ int3 __device__ INDEX3D(){
  int3 id;
  id.x     = _mul(blockIdx.x,blockDim.x) + threadIdx.x;
  id.y     = _mul(blockIdx.y ,blockDim.y) + threadIdx.y;
  id.z     = _mul(blockIdx.z ,blockDim.z) + threadIdx.z;
  return id;
}
/**
	@brief Returns the global thread index for 1D thread block.
	In Fermi architecture the 1D grid size 	is limited to 65535 blocks 
	and therefore when this is insufficient a 2D grid size is set.
*/
__forceinline__ uint __device__ INDEX1D(){
#if (__CUDA_ARCH__ >= 300)
	return blockIdx.x * blockDim.x + threadIdx.x;
#else
	uint id = gridDim.x * blockIdx.y  + blockIdx.x;
	return blockDim.x * id + threadIdx.x; 
#endif
}
#endif




}

#endif 
