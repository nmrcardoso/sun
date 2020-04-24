#include <cmath>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <culqcd.h>
#include "gnuplot.h"



#include <stdio.h>  // defines FILENAME_MAX
#include <unistd.h> // for getcwd()
#include <iostream>




using namespace std;
using namespace CULQCD;


template <class Real, ArrayType mygaugein>
void RunOnDeviceTEST(int argc, char** argv);

//NEEDS: export CULQCD_RESOURCE_PATH="path to folder where the tuning parameters are saved..."
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){



	cout << sizeof(curandStateMRG32k3a) << ":::::" << sizeof(curandStateXORWOW) << endl;
	cout << sizeof(float) << ":::::" << sizeof(double) << endl;


	COUT << "###############################################################################" << endl;
	const ArrayType mygauge = SOA;// SOA/SOA12/SOA8 for SU(3) and SOA for N>3
	RunOnDeviceTEST<double, mygauge>(argc, argv);
	COUT << "###############################################################################" << endl;
	EndCULQCD(0);
	COUT << "###############################################################################" << endl;
	exit(0);
}  
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
template <class Real, ArrayType mygaugein>
void RunOnDeviceTEST(int argc, char** argv){

	int ns = atoi(argv[1]);
	int nt = atoi(argv[2]);
	float beta0 = atof(argv[3]);
	int gpuid= atoi(argv[4]);
	int usetex = atoi(argv[5]); 
	PARAMS::UseTex = usetex;
	string filenamein = argv[6];
	int iter = atoi(argv[7]);




	//initCULQCD(gpuid, DEBUG_VERBOSE, TUNE_YES);
	initCULQCD(gpuid, SUMMARIZE, TUNE_YES);
	//initCULQCD(gpuid, VERBOSE, TUNE_YES);
	//if TUNE_YES user must set export CULQCD_RESOURCE_PATH="path to folder where the tuning parameters are saved..."
	//---------------------------------------------------------------------------------------
	// Create timer
	//---------------------------------------------------------------------------------------
	Timer t0;
	//---------------------------------------------------------------------------------------
	// Start timer
	//---------------------------------------------------------------------------------------
	t0.start(); 
	//---------------------------------------------------------------------------------------
	//---------------------------------------------------------------------------------------
	// Set Lattice Gauge Parameters and copy to Device constant memory
    // also sets some kernel launch parameters
    // SETPARAMS( bool tex reads on/off, beta, nx, ny, nz, nt);
	//---------------------------------------------------------------------------------------
	SETPARAMS(usetex, beta0, ns, ns, ns, nt, true);
	//TO ENABLE/DISABLE READS FROM TEXTURES anywhere in the code....!!!!!!
	//UseTextureMemory(true/false);

	gauge AA(mygaugein, Device, PARAMS::Volume * 4, true);
	AA.Details();
	
  	ReadBin_Gauge<Real, double>(AA, filenamein);

	PlaquetteCUB<Real> plaqCUB(AA);
	OnePolyakovLoop<Real> poly(AA);
    //Reunitarize container
	Reunitarize<Real> reu(AA);
	reu.Run();   //Reunitarize gauge field 
    
	//Calculate plaquette
	plaqCUB.Run();
	plaqCUB.printValue();
	//Calculate polyakov loop
	poly.Run();
	poly.printValue();
    //.stat() for performance printing
	plaqCUB.stat();
	poly.stat();






if(0){
	gauge BB(SOA, Device, PARAMS::Volume * 4, false);
	//copy the gauge field in even-odd order to normal order to use by ape and Wilson loop
	BB.Copy(AA);
	int Rmax = PARAMS::Grid[0] / 2;
    int Tmax = PARAMS::Grid[3] / 3;
    complex *res = (complex*) safe_malloc(sizeof(complex)* (Rmax+1)*(Tmax+1));

    //APE in space, this is not yet implemented in MPI
    //ApplyAPEinSpace(in, 0.2, 25, 20, 0.0000000001);
    //APE in time, this is not yet implemented in MPI
    //ApplyAPEinTime(in, 0.2, 1, 20, 0.0000000001);
    //Wilson loop, this is not yet implemented in MPI
    WilsonLoop(BB, res, Rmax, Tmax); //This version uses more device memory, but is faster
    //CWilsonLoop(in, res, Rmax, Tmax); 
	ofstream fileout;
	string filename = "WilsonLoop_" + ToString(iter) + ".dat";
    fileout.open(filename.c_str(), ios::out);
	fileout.precision(16);
	fileout.setf(ios_base::scientific);
    cout << "Writing data to: " << filename << endl;
    if (fileout.is_open()) {
		for(int r = 0; r < Rmax; r++){
			for(int it = 0; it < Tmax; it++){
		    	fileout << r << "\t" << it << "\t" << res[it + r * (Tmax+1)].real() << '\n';
			}
			fileout.flush();
		}
		fileout.close();
		//gnuplot3D(gp, 0, filename);
	}
	BB.Release();
    host_free(res);
}




if(1){


	RNG randstates;
	int rngSeed = 1234;
	rngSeed = (unsigned)time(0);


//	for(int ss=0;ss<5;ss++){
//	for(int ss=2;ss<3;ss++){
	for(int ss=0;ss<6;ss++){
Timer a0;
a0.start();
		string filename1 = "WilsonLoop_A0_" + ToString(iter) + ".dat";		
		gauge BB(SOA, Device, PARAMS::Volume * 4, false);
		//copy the gauge field in even-odd order to normal order to use by ape and Wilson loop
		BB.Copy(AA);
		if(ss==1){
			COUT << "#################################################################" << endl;
			//Apply APE Smearing
			ApplyAPEinSpace<Real>(BB, 0.2, 25, 10, 1.e-10); 
			COUT << "#################################################################" << endl;
			filename1 = "WilsonLoop_APE_S_A0_" + ToString(iter) + ".dat";
		}
		if(ss==2){
			randstates.Init(rngSeed);
			gauge DD(BB.Type(), Device, PARAMS::Volume * 4, false);	
			DD.Copy(BB);
			ApplyMultiHit(DD, BB, randstates, 100);
			//ApplyMultiHitExt(DD, BB, randstates, 100);
			DD.Release();
			filename1 = "WilsonLoop_MHIT_A0_" + ToString(iter) + ".dat";
		}
		if(ss==3){
			COUT << "#################################################################" << endl;
			//Apply APE Smearing
			ApplyAPEinTime<Real>(BB, 0.2, 1, 10, 1.e-10); 
			COUT << "#################################################################" << endl;
			filename1 = "WilsonLoop_APE_T_A0_" + ToString(iter) + ".dat";
		}
		if(ss==4){
			COUT << "#################################################################" << endl;
			//Apply APE Smearing
			ApplyAPEinTime<Real>(BB, 0.2, 1, 10, 1.e-10); 
			ApplyAPEinSpace<Real>(BB, 0.2, 25, 10, 1.e-10); 
			COUT << "#################################################################" << endl;
			filename1 = "WilsonLoop_APE_ST_A0_" + ToString(iter) + ".dat";
		}
		if(ss==5){
			randstates.Init(rngSeed);
			gauge DD(BB.Type(), Device, PARAMS::Volume * 4, false);	
			DD.Copy(BB);
			ApplyMultiHit(DD, BB, randstates, 100);
			//ApplyMultiHitExt(DD, BB, randstates, 100);
			DD.Release();
			ApplyAPEinSpace<Real>(BB, 0.4, 50, 10, 1.e-10); 
			filename1 = "WilsonLoop_APE_MHIT_A0_" + ToString(iter) + ".dat";
		}
		
		
if(1){
 	int Rmax = PARAMS::Grid[0] / 2;
    int Tmax = PARAMS::Grid[3] / 2;
    complex *res = (complex*) safe_malloc(sizeof(complex)* (Rmax+1)*(Tmax+1));

    WilsonLoop(BB, res, Rmax, Tmax); //This version uses more device memory, but is faster
//    CWilsonLoop(BB, res, Rmax, Tmax); 
//	nmeas++;
	ofstream fileout;
    fileout.open(filename1.c_str(), ios::out);
    cout << "Writing data to: " << filename1 << endl;
    if (fileout.is_open()) {
		for(int r = 0; r < Rmax; r++){
			for(int it = 0; it < Tmax; it++){
//				meas[it + r * (Tmax+1)] += res[it + r * (Tmax+1)];
//		    	fileout << r << "\t" << it << "\t" << meas[it + r * (Tmax+1)].real()/double(nmeas) << '\n';
		    	fileout << r << "\t" << it << "\t" << res[it + r * (Tmax+1)].real() << '\n';
				//if(it==Tmax-1) fileout << '\n';
			}
			fileout.flush();
		}
		fileout.close();
		//gnuplot3D(gp, 0, filename);
	}
	
    host_free(res);
}

		
		
		
		
	if(0){		
		Sigma_g_plus<Real>  arg(12, 16);


		ofstream fileout; 
		string filename1 = "WilsonLoop_A0_" + ToString(iter) + ".dat";
		if(ss==1) filename1 = "WilsonLoop_APE_A0_" + ToString(iter) + ".dat";
		if(ss==2) filename1 = "WilsonLoop_APE_MHIT_A0_" + ToString(iter) + ".dat";
		if(ss==3) filename1 = "WilsonLoop_Stout_A0_" + ToString(iter) + ".dat";
		if(ss==4) filename1 = "WilsonLoop_HYP_A0_" + ToString(iter) + ".dat";

		COUT << "Saving in file: " << filename1 << endl;

		fileout.open(filename1.c_str(), ios::out);
		fileout.precision(16);
		fileout.setf(ios_base::scientific);
		if (!(fileout.is_open())) {
		    COUT << "Error Writing Data File..." << filename1 << endl;	
			exit(0);
		}
		a0.start();
	  	int numdirs = 3;
		for(int r = 2; r <= arg.Rmax; r++){
			COUT << "###############################################################################" << endl;
			COUT << "Calculating radius: " << r << endl;
			COUT << "###############################################################################" << endl;
			CUDA_SAFE_CALL(cudaMemset(arg.wloop, 0,  arg.wloop_size));
			for(int mu=0; mu < numdirs; mu++){
		        CalcWLOPs_A0<Real>(BB, &arg, r, mu);
				CalcWilsonLoop_A0<Real>(BB, &arg, r, mu);
			}
	  		CUDA_SAFE_CALL(cudaMemcpy(arg.wloop_h, arg.wloop, arg.wloop_size, cudaMemcpyDeviceToHost));	
			//Normalize
			for(int iop = 0; iop < arg.totalOpN; iop++)
			for(int it = 0; it <= arg.Tmax; it++){
				int id = it + iop * (arg.Tmax+1);
				arg.wloop_h[id] /= (Real)(PARAMS::Volume * numdirs * 3);
			}
			for(int it = 0; it <= arg.Tmax; it++){
				fileout << r << "\t" << it;
				for(int iop = 0; iop < arg.totalOpN; iop++){
					int id = it + iop * (arg.Tmax+1);
			        fileout << "\t" << arg.wloop_h[id] << "\t";
				}
				fileout << endl;
			}
		}
		fileout.close();
	}
	a0.stop();
	COUT << "###############################################################################" << endl;
	COUT << "Time A0: " << a0.getElapsedTime() << " s" << endl; 
	COUT << "###############################################################################" << endl;
	BB.Release();
}
	randstates.Release();






}
    AA.Release();
	COUT << "###############################################################################" << endl;
	COUT << "Time: " << t0.getElapsedTime() << " s" << endl; 
	COUT << "###############################################################################" << endl;
	return ;
}












