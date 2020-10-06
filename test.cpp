#include <cmath>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <culqcd.h>
#include <io_gauge.h>

using namespace std;
using namespace CULQCD;

      
template <class Real, ArrayType mygaugein, int actiontype>
void FinalRUN(int argc, char** argv);


template <class Real, ArrayType mygaugein>
void RestartRUN(int argc, char** argv);

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){
	COUT << "###############################################################################" << endl;
	if( atoi(argv[8] ) == 0)
		FinalRUN<float, SOA12, 0>(argc, argv);
	if( atoi(argv[8] ) == 1)
		FinalRUN<float, SOA12, 1>(argc, argv);
	if( atoi(argv[8] ) == 2)
		FinalRUN<float, SOA12, 2>(argc, argv);
	//RestartRUN<double, SOA12>(argc, argv);
	COUT << "###############################################################################" << endl;
	EndCULQCD(0);
	COUT << "###############################################################################" << endl;
	exit(0);
}  
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
template <class Real, ArrayType mygaugein, int actiontype>
void FinalRUN(int argc, char** argv){   

	int ns = atoi(argv[1]);
	int nt = atoi(argv[2]);
	float beta0 = atof(argv[3]);
	int maxsteps= atoi(argv[4]);
	int gpuid= atoi(argv[5]);
	int nmonte = 4;
	int nover = 7;
	int usetex = atoi(argv[6]); 
	double aniso = atof(argv[7]); 
	PARAMS::UseTex = usetex;
	
	//initCULQCD(gpuid, DEBUG_VERBOSE, TUNE_YES);
	initCULQCD(gpuid, DEBUG_VERBOSE, TUNE_YES);

	int verbosemonte = 10;	
	int termalizationiterterm = 450;//990;//50;
	int stepstomeasures = 50;
	
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
  SETPARAMS(usetex, beta0, ns, ns, ns, nt, true, aniso);
  //TO ENABLE/DISABLE READS FROM TEXTURES anywhere in the code....!!!!!!
  //UseTextureMemory(true/false);
  
	COUT << "Using action type: " << actiontype << endl;
	cout << "Anisotropy factor: " << aniso << endl;


	RNG randstates;
	int rngSeed = 1234;
	//rngSeed = (unsigned)time(0);
	gauge AA(mygaugein, Device, PARAMS::Volume * 4, true);	
	
	
	AA.Details();
	randstates.Init(rngSeed);
	AA.Init(randstates); 
	
	
/*	Lattice Monte Carlo Step... 10
:::::0.819262::::1
Plaquette: < 4.506879746914e-01 : 8.029695749283e-01 > :: mean: 6.268287897110e-01
Polyakov Loop: < 2.124324906617e-03 : -3.250153968111e-03 : 3.882815595716e-03 >
HeatBath:  0.401647 s	12.9982 GB/s	44.4475 GFlops
OverRelaxation:  0.386388 s	12.2832 GB/s	46.7798 GFlops

	
:::::0.819236::::1
Plaquette: < 4.510477185249e-01 : 8.029575943947e-01 > :: mean: 6.270026564598e-01
Polyakov Loop: < -8.430885500275e-04 : -1.698338543065e-03 : 1.896088593639e-03 >
HeatBath:  0.40277 s	12.962 GB/s	44.3236 GFlops
OverRelaxation:  0.387157 s	12.2588 GB/s	46.6869 GFlops
Reunitarize:  0.002165 s	219.219 GB/s	411.923 GFlops
Plaquette:  0.006647 s	199.331 GB/s	646.881 GFlops
Polyakov Loop:  0.000302 s	196.785 GB/s	870.271 GFlops
Lattice Monte Carlo Step... 20
:::::0.81941::::1
Plaquette: < 4.506992399693e-01 : 8.029869794846e-01 > :: mean: 6.268430948257e-01
Polyakov Loop: < -8.479307871312e-05 : -7.071811705828e-04 : 7.122464594431e-04 >
HeatBath:  0.407151 s	12.8225 GB/s	43.8466 GFlops
OverRelaxation:  0.391658 s	12.118 GB/s	46.1504 GFlops
Reunitarize:  0.002182 s	217.511 GB/s	408.714 GFlops
*/
	
	PlaquetteCUB<Real> plaq(AA);
	//Plaquette<Real> plaq(AA);
	OnePolyakovLoop<Real> poly(AA);
	HeatBath<Real, actiontype> heat(AA, randstates);
	OverRelaxation<Real, actiontype> over(AA);
	Reunitarize<Real> reu(AA);
	//reu.Run();        
	complex plaq_value = plaq.Run();
	poly.Run();
	plaq.printValue();
	poly.printValue();
	plaq.stat();
	poly.stat();
	//Thermatization!!!!!!!!!!!!!!!!!!!!	
	for(int step = 1; step <= termalizationiterterm; step++){
		if(step%verbosemonte == 0 || step == termalizationiterterm) COUT << "Lattice Monte Carlo Step... " << step << endl;
		
		if( actiontype == 1 || actiontype == 2) {
			double ass = sqrt(sqrt(plaq_value.real()));
			double att = sqrt(plaq_value.imag())/ass;
			if(step==1) { ass = 0.613; att = 0.613;}
			att=1.0;
			if(step%verbosemonte == 0)  
				cout << ":::::" << ass << "::::" << att << endl;
			SetUsUt(ass, att);
		}
		for(int i = 0; i< nmonte; i++) heat.Run();
		for(int i = 0; i< nover; i++)  over.Run();
		reu.Run();
		plaq_value = plaq.Run();
		poly.Run();
		if(step%verbosemonte == 0 || step == termalizationiterterm) {
			plaq.printValue();
			poly.printValue();
			heat.stat();
			over.stat();
			reu.stat();
			plaq.stat();
			poly.stat();
		}
	}
	COUT << "Time: " << t0.getElapsedTime() << " s" << endl; 
	COUT << "###############################################################################" << endl;
	//////////////////////////////////////////////////////////////////////////////////////////////
	for(int step = termalizationiterterm+1; step <= maxsteps; step++){
		if(step%verbosemonte == 0 || step == maxsteps) COUT << "Lattice Monte Carlo Step... " << step << endl;
		
		if( actiontype == 1 || actiontype == 2) {
			double ass = sqrt(sqrt(plaq_value.real()));
			double att = sqrt(plaq_value.imag())/ass;
			att=1.0;
			if(step%verbosemonte == 0)  cout << ":::::" << ass << "::::" << att << endl;
			SetUsUt(ass, att);
		}
		for(int i = 0; i< nmonte; i++) heat.Run();
		for(int i = 0; i< nover; i++)  over.Run();
		reu.Run();
		plaq_value = plaq.Run();
		poly.Run();
		if(step%verbosemonte == 0 || step == maxsteps) {
			plaq.printValue();
			poly.printValue();
			heat.stat();
			over.stat();
			reu.stat();
			plaq.stat();
			poly.stat();
		}
		if(step%stepstomeasures == 0){
			COUT << "=====================================================================================" << endl;
			COUT << "=====================================================================================" << endl;
			COUT << "=====================================================================================" << endl;
			string fileout = "SU3_" + ToString(PARAMS::Grid[0]) + "_" + ToString(PARAMS::Grid[1]) + "_"; 
			fileout += ToString(PARAMS::Grid[2]) + "_" + ToString(PARAMS::Grid[3]) + "_" + ToString(PARAMS::Beta) + "_"; 
			fileout += ToString(step) + "_PHB" + ToString(nmonte) + "_OR" + ToString(nover) + "_dp.bin"; 
			
			//SaveBin_Gauge<Real, double>(AA, fileout, true);
			fileout = "rng_state_" + ToString(step) + ".bin";
			//randstates.Save(fileout);
			
			for(int meas = 1; meas < 3; ++meas){
				gauged BB0(SOA, Device, PARAMS::Volume * 4, true);
				GaugeCopy(AA, BB0);
				gauged BB(SOA, Device, PARAMS::Volume * 4, false);
				BB.Copy(BB0);
				BB0.Release();	
				Reunitarize<double> reuniteg(BB);
				reuniteg.Run();  
				
				string meastype = "";
				if(meas==20){
					gauged DD(BB.Type(), Device, PARAMS::Volume * 4, false);	
					DD.Copy(BB);
					ApplyMultiHit(DD, BB, randstates, 100, actiontype);
					DD.Release();
					meastype += "_mhit";
				}
				//if(meas>0){
				if(meas==1){
					ApplyAPEinSpace<double>(BB, 0.4, 50);
					//ApplyAPEinSpace<double>(BB, 0.1, 20);
					meastype += "_ape";
				}
				if(meas==2){
				//if(meas==0){
					ApplyStoutinSpace<double>(BB, 0.15, 20);
					//ApplyAPEinSpace<double>(BB, 0.1, 20);
					meastype += "_stout";
				}
				
				if(meas==111){
					//ApplyAPEinSpace<double>(BB, 0.1, 10);
					ApplyAPE2inSpace<double>(BB, 0.1, 10);
					meastype += "_ape2";
				}
				
				int Rmax = PARAMS::Grid[0] / 2;
				int Tmax = PARAMS::Grid[3] / 2;
				if(Tmax > 24) Tmax = 24;
				//if(Tmax > 24) Tmax = 16;
				
			   
			   	complexd *res;
				if(Tmax > Rmax)
				 res = (complexd*) safe_malloc(sizeof(complexd)* (Rmax+1)*(Tmax+1));
				else
				 res = (complexd*) safe_malloc(sizeof(complexd)* (Rmax+1)*(Rmax+1));
				 
				 
				string filename = ToString(PARAMS::Grid[0]) + "_" + ToString(PARAMS::Grid[1]) + "_"; 
				filename += ToString(PARAMS::Grid[2]) + "_" + ToString(PARAMS::Grid[3]) + "_" + ToString(PARAMS::Beta) + "_" + ToString(aniso) + "_"  + ToString(actiontype) + "_"; 
				filename += ToString(step) + "_PHB" + ToString(nmonte) + "_OR" + ToString(nover) + "_dp.bin";
				 
				 
				 
				 
				 
				 if(1){		
				 	Timer a0;
					a0.start();
					Sigma_g_plus<double>  arg(Rmax, Tmax);


					ofstream fileout; 
					string filename1 = "WilsonLoop_A2_" + meastype + "_";
					/*if(meas==1) filename1 = "WilsonLoop_APE_A2_";
					if(meas==2) filename1 = "WilsonLoop_APE_MHIT_A2_";*/
					 filename1 += filename;
					

					COUT << "Saving in file: " << filename1 << endl;

					fileout.open(filename1.c_str(), ios::out);
					fileout.precision(16);
					fileout.setf(ios_base::scientific);
					if (!(fileout.is_open())) {
						COUT << "Error Writing Data File..." << filename1 << endl;	
						exit(0);
					}
				  	int numdirs = 3;
					for(int r = 1; r <= arg.Rmax; r++){
						COUT << "###############################################################################" << endl;
						COUT << "Calculating radius: " << r << endl;
						COUT << "###############################################################################" << endl;
						CUDA_SAFE_CALL(cudaMemset(arg.wloop, 0,  arg.wloop_size));
						for(int mu=0; mu < numdirs; mu++){
							CalcWLOPs_A0<double>(BB, &arg, r, mu);
							CalcWilsonLoop_A0<double>(BB, &arg, r, mu);
						}
				  		CUDA_SAFE_CALL(cudaMemcpy(arg.wloop_h, arg.wloop, arg.wloop_size, cudaMemcpyDeviceToHost));	
						//Normalize
						for(int iop = 0; iop < arg.totalOpN; iop++)
						for(int it = 0; it <= arg.Tmax; it++){
							int id = it + iop * (arg.Tmax+1);
							arg.wloop_h[id] /= (double)(PARAMS::Volume * numdirs * 3);
						}
						for(int it = 0; it <= arg.Tmax; it++){
							fileout << r << "\t" << it;
							for(int iop = 0; iop < arg.totalOpN; iop++){
								int id = it + iop * (arg.Tmax+1);
								fileout << "\t" << arg.wloop_h[id] << "\t";
								//if(iop==5) cout << r << "\t" << it  << "\t" << arg.wloop_h[id] << endl; 
							}
							fileout << endl;
						}
					}
					fileout.close();
					a0.stop();
					COUT << "###############################################################################" << endl;
					COUT << "Time A0: " << a0.getElapsedTime() << " s" << endl; 
					COUT << "###############################################################################" << endl;
				}
				 
				 
				 
				 
				 
				 
				if(0){
				WilsonLoop(BB, res, Rmax, Tmax); //This version uses more device memory, but is faster
				//CWilsonLoop(in, res, Rmax, Tmax); 
				string filename = "WL2_aniso_" + ToString(PARAMS::Grid[0]) + "_" + ToString(PARAMS::Grid[1]) + "_"; 
				filename += ToString(PARAMS::Grid[2]) + "_" + ToString(PARAMS::Grid[3]) + "_" + ToString(PARAMS::Beta) + "_" + ToString(aniso) + "_"  + ToString(actiontype) + "_"; 
				filename += ToString(step) + "_PHB" + ToString(nmonte) + "_OR" + ToString(nover) + meastype + "_dp.bin";
				ofstream fileWL;
				fileWL.open(filename.c_str(), ios::out);
				fileWL.precision(16);
				fileWL.setf(ios_base::scientific);
				cout << "Writing data to: " << filename << endl;
				if (fileWL.is_open()) {
					for(int r = 0; r <= Rmax; r++){
						for(int it = 0; it <= Tmax; it++){
							fileWL << r << "\t" << it << "\t" << res[it + r * (Tmax+1)].real() << '\n';
							cout << r << "\t" << it << "\t" << res[it + r * (Tmax+1)].real() << '\n';
						}
						fileWL.flush();
					}
					fileWL.close();
				}
				}
				
				
				if(0){
				Tmax = Rmax;
				WilsonLoopSS(BB, res, Rmax, Tmax); //This version uses more device memory, but is faster
				//CWilsonLoop(in, res, Rmax, Tmax); 
				string filename = "WL2SS_aniso_" + ToString(PARAMS::Grid[0]) + "_" + ToString(PARAMS::Grid[1]) + "_"; 
				filename += ToString(PARAMS::Grid[2]) + "_" + ToString(PARAMS::Grid[3]) + "_" + ToString(PARAMS::Beta) + "_" + ToString(aniso) + "_"  + ToString(actiontype) + "_"; 
				filename += ToString(step) + "_PHB" + ToString(nmonte) + "_OR" + ToString(nover) +  meastype + "_dp.bin";
				ofstream fileWL;
				fileWL.open(filename.c_str(), ios::out);
				fileWL.precision(16);
				fileWL.setf(ios_base::scientific);
				cout << "Writing data to: " << filename << endl;
				if (fileWL.is_open()) {
					for(int r = 0; r <= Rmax; r++){
						for(int it = 0; it <= Tmax; it++){
							fileWL << r << "\t" << it << "\t" << res[it + r * (Tmax+1)].real() << '\n';
							//cout << r << "\t" << it << "\t" << res[it + r * (Tmax+1)].real() << '\n';
						}
						fileWL.flush();
					}
					fileWL.close();
				}
				}	
				
				
				if(0){
				Tmax = Rmax;
				CWilsonLoopSS(BB, res, Rmax, Tmax); //This version uses more device memory, but is faster
				//CWilsonLoop(in, res, Rmax, Tmax); 
				string filename = "CWL1SS_aniso_" + ToString(PARAMS::Grid[0]) + "_" + ToString(PARAMS::Grid[1]) + "_"; 
				filename += ToString(PARAMS::Grid[2]) + "_" + ToString(PARAMS::Grid[3]) + "_" + ToString(PARAMS::Beta) + "_" + ToString(aniso) + "_"  + ToString(actiontype) + "_"; 
				filename += ToString(step) + "_PHB" + ToString(nmonte) + "_OR" + ToString(nover) +  meastype + "_dp.bin";
				ofstream fileWL;
				fileWL.open(filename.c_str(), ios::out);
				fileWL.precision(16);
				fileWL.setf(ios_base::scientific);
				cout << "Writing data to: " << filename << endl;
				if (fileWL.is_open()) {
					for(int r = 0; r <= Rmax; r++){
						for(int it = 0; it <= Tmax; it++){
							fileWL << r << "\t" << it << "\t" << res[it + r * (Tmax+1)].real() << '\n';
							//cout << r << "\t" << it << "\t" << res[it + r * (Tmax+1)].real() << '\n';
						}
						fileWL.flush();
					}
					fileWL.close();
				}
				}			
				BB.Release();
				host_free(res);
			}
			//return;
		}
	}
	randstates.Release();
	AA.Release();
	COUT << "###############################################################################" << endl;
	COUT << "Time: " << t0.getElapsedTime() << " s" << endl; 
	COUT << "###############################################################################" << endl;
	return ;
}



//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
template <class Real, ArrayType mygaugein>
void RestartRUN(int argc, char** argv){   

	
	int ns = atoi(argv[1]);
	int nt = atoi(argv[2]);
	float beta0 = atof(argv[3]);
	int gpuid= atoi(argv[4]);
	int usetex = atoi(argv[5]); 
	PARAMS::UseTex = usetex;
	string filenamein = argv[6];
	int iter = atoi(argv[7]);
	string filename_rng = argv[8];
	int maxsteps = atoi(argv[9]);
	int nmonte = 4;
	int nover = 7;
	

	//initCULQCD(gpuid, DEBUG_VERBOSE, TUNE_YES);
	initCULQCD(gpuid, VERBOSE, TUNE_YES);

	int verbosemonte = 10;	
	int stepstomeasures = 100;
	
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






	RNG randstates;
	randstates.Init(1234); //Need this only to allocate... I didn't create a separate option if read state!!!!
	randstates.Read(filename_rng);
	gauge AA(mygaugein, Device, PARAMS::Volume * 4, true);
	AA.Details();
	
  	ReadBin_Gauge<Real, double>(AA, filenamein, true);

	
	PlaquetteCUB<Real> plaq(AA);
	OnePolyakovLoop<Real> poly(AA);
	HeatBath<Real, 0> heat(AA, randstates);
	OverRelaxation<Real, 0> over(AA);
	Reunitarize<Real> reu(AA);
	//reu.Run();   
    
	//Calculate plaquette
	plaq.Run();
	plaq.printValue();
	//Calculate polyakov loop
	poly.Run();
	poly.printValue();
    //.stat() for performance printing
	plaq.stat();
	poly.stat();
	
	//////////////////////////////////////////////////////////////////////////////////////////////
	for(int step = iter+1; step <= maxsteps; step++){
		if(step%verbosemonte == 0 || step == maxsteps) COUT << "Lattice Monte Carlo Step... " << step << endl;
		for(int i = 0; i< nmonte; i++) heat.Run();
		for(int i = 0; i< nover; i++)  over.Run();
		reu.Run();
		plaq.Run();
		poly.Run();
		if(step%verbosemonte == 0 || step == maxsteps) {
			plaq.printValue();
			poly.printValue();
			heat.stat();
			over.stat();
			reu.stat();
			plaq.stat();
			poly.stat();
		}
		if(step%stepstomeasures == 0){
			COUT << "=====================================================================================" << endl;
			COUT << "=====================================================================================" << endl;
			COUT << "=====================================================================================" << endl;
			string fileout = "SU3_" + ToString(PARAMS::Grid[0]) + "_" + ToString(PARAMS::Grid[1]) + "_"; 
			fileout += ToString(PARAMS::Grid[2]) + "_" + ToString(PARAMS::Grid[3]) + "_" + ToString(PARAMS::Beta) + "_"; 
			fileout += ToString(step) + "_PHB" + ToString(nmonte) + "_OR" + ToString(nover) + "_dp.bin"; 
			
			SaveBin_Gauge<Real, double>(AA, fileout, true);
			fileout = "rng_state_" + ToString(step) + ".bin";
			randstates.Save(fileout);
		}
	}
	randstates.Release();
	AA.Release();
	COUT << "###############################################################################" << endl;
	COUT << "Time: " << t0.getElapsedTime() << " s" << endl; 
	COUT << "###############################################################################" << endl;
	return ;
}



