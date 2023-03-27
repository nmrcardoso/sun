
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



template<class Real>
void WilsonLoopSS1(gauge array, gauge array_nosmear, complex *res, int radius);



template<class Real>
void WilsonLoopR(gauge array, gauge array_nosmear, complex *res, int radius[3], int Tmax, int mu, int nu);
}

#endif

