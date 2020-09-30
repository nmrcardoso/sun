
#ifndef CWILSON_LOOP_H
#define CWILSON_LOOP_H

#include <constants.h>


namespace CULQCD{



template<class Real>
void CWilsonLoop(gauge array, complex *res, int radius, int Tmax);


template<class Real>
void WilsonLoop(gauge array, complex *res, int radius, int Tmax);




template<class Real>
void CWilsonLoopSS(gauge array, complex *res, int radius, int Tmax);

template<class Real>
void WilsonLoopSS(gauge array, complex *res, int radius, int Tmax);
}

#endif

