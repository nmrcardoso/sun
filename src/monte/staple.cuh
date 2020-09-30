#ifndef STAPLE_KERNEL_CUH
#define STAPLE_KERNEL_CUH

#include <gaugearray.h>

using namespace std;


namespace CULQCD{


template <class Real> 
void CalculateStaple(gauge array, int oddbit, int mu, int actiontype);


template <class Real>
gauge* GetStapleArray();

void FreeStapleArray();
}

#endif
