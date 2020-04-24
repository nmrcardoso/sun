/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
 
#ifndef __REDUCTION_CUH__
#define __REDUCTION_CUH__



template <class T>
void call_reduce(int size, int threads, int blocks, 
                 int whichKernel, T *d_idata, T *d_odata);
template <class T>
void call_reduce_wstream(int size, int threads, int blocks, 
                         int whichKernel, T *d_idata, T *d_odata, const cudaStream_t &stream);
    

#endif
