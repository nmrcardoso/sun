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
#include <vector>
#include <rwdirectory.h>
#include <labeling.h>
#include <map>
using namespace std;
using namespace CULQCD;

      
template <class Real, ArrayType mygaugein, int actiontype>
void FinalRUN(int argc, char** argv);


template <class Real, ArrayType mygaugein,int actiontype>
void RestartRUN(int argc, char** argv);

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){
    COUT << "##############################arguments= "<<argc-1<<"##########################################" << endl;
    if(argc==11 or argc==14){
    if(atoi( argv[8] ) == 0)
        FinalRUN<float, SOA, 0>(argc, argv);
    if(atoi(argv[8] ) == 1)
        FinalRUN<float, SOA, 1>(argc, argv);
    if(atoi(argv[8] ) == 2)
        FinalRUN<float, SOA, 2>(argc, argv);
  }else{
   			cout<< "number of the argv should 10 or 13 parameters!"<<endl;
    		exit(EXIT_FAILURE);
   }
    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
//    if(argc==14 and atoi(argv[8] ) == 0)
//        RestartRUN<float, SOA12, 0 >(argc, argv);
//    if(argc==14 and atoi(argv[8] ) == 1)
//        RestartRUN<float, SOA12, 1 >(argc, argv);
//    if(argc==14 and atoi(argv[8] ) == 2)
//        RestartRUN<float, SOA12, 2 >(argc, argv);
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


	cout<< "############ cpp version ="+to_string(__cplusplus)+"##############"<<endl;
    int ns = atoi(argv[1]);
    int nt = atoi(argv[2]);
    float beta0 = atof(argv[3]);
    int maxsteps= atoi(argv[4]);
    int gpuid= atoi(argv[5]);
    int nmonte = 4;
    int nover = 7;
    int usetex = atoi(argv[6]); 
    double aniso = atof(argv[7]); 
    //argv[8] is type of the action
    PARAMS::UseTex = usetex;
    string config_dir=argv[9];
    string results_dir=argv[10];
    cout<<"#########################"<< currentDateTime() <<"################################"<<endl;
//    directory::make_nested_folder(config_dir);
//    directory::make_nested_folder(results_dir);
    config_dir+="/";
    string cmd="mkdir -vp "+config_dir;
    system(cmd.c_str());
    cmd="mkdir -vp "+results_dir;
    system(cmd.c_str());
    //results_dir+="/";
    bool remove_prv_config=true;
    bool config_header=true;
    printf(" ################################# creating a backup ####################################\n");
    string cur_time= currentDateTime();
//    string backup_name=results_dir+"/code_backup_"+cur_time+".tar.gz";
    string backup_name="/code_backup_"+cur_time+".tar.gz";
    cout<<backup_name<<endl;
    cmd="tar cfz "+results_dir+backup_name+" *.cpp  include/* src/* Makefile ";
    system(cmd.c_str());
    printf(" a backup of code has been made\n");
    printf(" maximum step set to %d\n", maxsteps);


    initCULQCD(gpuid, DEBUG_VERBOSE, TUNE_YES);
    
    //initCULQCD(gpuid, DEBUG_VERBOSE, TUNE_YES);
    initCULQCD(gpuid, DEBUG_VERBOSE, TUNE_YES);
    
    int verbosemonte = 10;    
    int termalizationiterterm = 500;//990;//50;
    int stepstomeasures = 50;
    string config_precision="";
    
    
    if (sizeof(Real)==8){
        config_precision="_dp";
    }else if(sizeof(Real)==4){
        config_precision="_sp";
    }
    bool save_config=1;

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
    RNG randstates_mhit;
    int rngSeed = 1234;
    //rngSeed = (unsigned)time(0);
    gauge AA(mygaugein, Device, PARAMS::Volume * 4, true);   
    
     
    AA.Details();
    COUT<< "For lattice the randstates is being set::"<<endl;
    randstates.Init(rngSeed);
    
    COUT<< "For Multi-hit the randstates is being set::"<<endl;
    randstates_mhit.Init(rngSeed);
    
    AA.Init(randstates); 
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
    
    double asss=0.82006;
	double attt=1.0;
	SetUsUt(asss, attt);	
    //Thermatization!!!!!!!!!!!!!!!!!!!!    
    if (argc==11){
        printf("the number of arguments is %d, the process start from beginning!\n", argc);
    for(int step = 1; step <= termalizationiterterm; step++){
        if(step%verbosemonte == 0 || step == termalizationiterterm) COUT << "Lattice Monte Carlo Step... " << step << endl;
        
        if(0)if( actiontype == 1 || actiontype == 2) {
            double ass = sqrt(sqrt(plaq_value.real()));
            double att = sqrt(plaq_value.imag())/ass;
            if(step==1) { ass = 0.613; att = 0.613;}
            att=1.0;
            if(step%verbosemonte == 0)  
                cout << "::ass:::" << ass << "::att::" << att << endl;
            SetUsUt(ass, att);
        }// these value first set by nuno
        
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
    // this is the end of first thermalization
    COUT << "Time for Thermatization: " << t0.getElapsedTime() << " s" << endl; 
    COUT << "############################### First thermalization #################################" << endl;
   }
    else if(argc==14){// this part is to restart from previous saved configuration
        printf("the number of arguments is %d, the process start from !\n", argc);
    
    string filename_rng=argv[11];
    string filenamein=argv[12];
    int iter=atoi(argv[13]);
    printf("iter set to %d\n", iter);
    
	termalizationiterterm = iter;
	int step = termalizationiterterm;


	if (config_header){
	    printf("your are working with configuration with header.\n");
        ReadBin_Gauge<Real, Real >(AA, filenamein, true);
    }else{
        printf("this configuration does not have header!\n");
        ReadBin_Gauge<Real, Real>(AA, filenamein, false);
    }
    
	randstates.Read(filename_rng);
	reu.Run();
	plaq.Run();
	poly.Run();
	plaq.printValue();
	poly.printValue();
    }


    //////////////////////////////////////////////////////////////////////////////////////////////
    Timer tEachConfig;
    tEachConfig.start();
    for(int step = termalizationiterterm+1; step <= maxsteps; step++){
        if(step%stepstomeasures==0) tEachConfig.start();
        if(step%verbosemonte == 0 || step == maxsteps) COUT << "Lattice Monte Carlo Step... " << step << endl;
        
        if(0)if( actiontype == 1 || actiontype == 2) {
            double ass = sqrt(sqrt(plaq_value.real()));
            double att = sqrt(plaq_value.imag())/ass;
            att=1.0;
            if(step%verbosemonte == 0)  cout << ":::::" << ass << "::::" << att << endl;
            SetUsUt(ass, att);
        }// these values first set by nuno
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
            string fileout_config = config_dir+"SU"+to_string(NCOLORS)+"_" + ToString(PARAMS::Grid[0]) + "_" + ToString(PARAMS::Grid[1]) + "_"; 
            fileout_config += ToString(PARAMS::Grid[2]) + "_" + ToString(PARAMS::Grid[3]) + "_B" + ToString(PARAMS::Beta) +
             "_X" + ToString(aniso) + "_A"  + ToString(actiontype)+"_S"+ToString(stepstomeasures)+"_";
             
            string fileout_prv_config=fileout_config;// this is to save previous config name
            
            fileout_config += ToString(step) + "_PHB" + ToString(nmonte) + "_OR" + ToString(nover) + config_precision+".bin"; 
            if (save_config)SaveBin_Gauge<Real, Real>(AA, fileout_config, true);
            
            int iter_prv=step-stepstomeasures;
            fileout_prv_config += ToString(iter_prv) + "_PHB" + ToString(nmonte) + "_OR" + ToString(nover) + config_precision+".bin"; 


            string fileout_rng =config_dir+ "rng_state_" + ToString(PARAMS::Grid[0]) + "_" + ToString(PARAMS::Grid[1]) + "_";
            
            fileout_rng+=ToString(PARAMS::Grid[2]) + "_" + ToString(PARAMS::Grid[3]) + "_B" + ToString(PARAMS::Beta);
            
            fileout_rng+= "_X" + ToString(aniso) + "_A"  + ToString(actiontype)+"_S"+ToString(stepstomeasures)+"_";
            
            string fileout_rng_prv=fileout_rng;
            
            fileout_rng+=ToString(step)+config_precision+".bin";
            //////////////////////////////
            if(save_config)randstates.Save(fileout_rng);
            //////////////////////////////
            fileout_rng_prv += ToString(iter_prv) + config_precision+".bin"; 
            cout<< "time for this configuration ::"<< tEachConfig.getElapsedTime()<<" (s)"<<endl;
            enum smearing{no_smearing,mhit, stout, ape_s, ape_t, hyp, mhit_ext, ape2_s,mhit_stout, hyp_ape, mhit_ext_ape,mhit_ape};
            /*#########################################*/
            vector<int> smearing_type={mhit_stout};
            /*#########################################*/
            for(int meas:smearing_type){
            //for(int meas = 1; meas < 3; ++meas){
                // notice calculation is done on BB, which is updated at the beginning of each iteration
//                gauge BB(SOA, Device, PARAMS::Volume * 4, false);
//                BB.Copy(AA);
                //this part need to be uncomment for Real=float
                gauged BB0(SOA, Device, PARAMS::Volume * 4, true);
                GaugeCopy(AA, BB0);
                
                
                gauged BB(SOA, Device, PARAMS::Volume * 4, false);
                BB.Copy(BB0);
                BB0.Release();
                
                Reunitarize<double> reuniteg(BB);
                reuniteg.Run();  
                string meastype = "";
                if(meas==mhit){
                    int n=100;
                    gauged DD(BB.Type(), Device, PARAMS::Volume * 4, false);    
                    DD.Copy(BB);
                    ApplyMultiHit(DD, BB, randstates_mhit, n, actiontype);
                    DD.Release();
                    meastype += "_mhit_"+ToString(n);
                }
                if(meas==mhit_ape){
                    int n_mhit=100;
                    gauged DD(BB.Type(), Device, PARAMS::Volume * 4, false);    
                    DD.Copy(BB);
                    ApplyMultiHit(DD, BB, randstates_mhit, n_mhit, actiontype);// this was in the change nuno make
                    DD.Release();
                    float alpha_s=0.4;
                    int nos=50;
                    ApplyAPEinSpace<double>(BB, alpha_s, nos, 10, 1.e-10);
                    meastype+="_mhit_"+ToString(n_mhit)+"_ape_al_"+ToString(alpha_s)+"_s_"+ToString(nos);
                    //al=alpha, st=step
                }
                //if(meas>0){
                if(meas==ape_s){
                    float alpha_s=0.4;
                    int nos=20;// number of step
                    ApplyAPEinSpace<double>(BB, alpha_s,nos );
                    //ApplyAPEinSpace<double>(BB, 0.1, 20);
                    meastype += "_ape_a_"+ToString(alpha_s)+"_s_"+ToString(nos);
                }
                if(meas==stout){
                //if(meas==0){
                    float alpha_s=0.15;
                    int nos=20;
                    ApplyStoutinSpace<double>(BB, alpha_s, nos);
                    //ApplyAPEinSpace<double>(BB, 0.1, 20);
                    meastype += "_stout_a_"+ToString(alpha_s)+"_s_"+ToString(nos);
                }
                
                if(meas==ape2_s){
                    float alpha_s=0.1;
                    int nos=10;
                    //ApplyAPEinSpace<double>(BB, 0.1, 10);
                    ApplyAPE2inSpace<double>(BB, alpha_s, nos);
                    meastype += "_ape2_a_"+ToString(alpha_s)+"_s_"+ToString(nos);
                }
                if(meas==mhit_stout){
                    int n_mhit=100;
                    gauged DD(BB.Type(), Device, PARAMS::Volume * 4, false);    
                    DD.Copy(BB);
                    ApplyMultiHit(DD, BB, randstates_mhit, n_mhit,actiontype);
                    DD.Release();
                    float alpha_s=0.15;
                    int nos=20;
                    ApplyStoutinSpace<double>(BB, alpha_s, nos);
                    meastype+="_mhit_"+ToString(n_mhit)+"_stout_al_"+ToString(alpha_s)+"_nos_"+ToString(nos);
                    //al=alpha, st=step
                }
                if(meas==hyp_ape){
                    ParamHYP HYPArg;
                    int n_hyp=1;
                    ApplyHYPinTime<double>(BB, n_hyp, 20, 1.e-10, HYPArg);// 20 is nhit
                    float alpha_s=0.5;
                    int nos=20;
                    ApplyAPEinSpace<double>(BB, alpha_s, nos, 10, 1.e-10); // 10 is nhit
                    meastype="_hyp_"+ToString(n_hyp)+"_ape_al_"+ToString(alpha_s)+"_nos_"+ToString(nos);
                }
                if(meas==mhit_ext_ape){
                    int nhit=100;
                    gauged DD(BB.Type(), Device, PARAMS::Volume * 4, false);
                    DD.Copy(BB);
                    ApplyMultiHitExt(DD, BB, randstates_mhit, nhit);
                    DD.Release();
                    float alpha=0.4;
                    int nos=20;
                    ApplyAPEinSpace<double>(BB, alpha, nos, 10, 1.e-10); 
                    meastype="_mhitEx_"+ToString(nhit)+"_ape_"+ToString(alpha)+"_nos_"+ToString(nos);
                }
                int Rmax = PARAMS::Grid[0] / 2;
                int Tmax = PARAMS::Grid[3] / 3;
                if(Tmax >24) Tmax = 24;
                if(Tmax<=16) Tmax=18;
                string filename = ToString(PARAMS::Grid[0]) + "_" + ToString(PARAMS::Grid[1]) + "_"; 
                filename += ToString(PARAMS::Grid[2]) + "_" + ToString(PARAMS::Grid[3]) + "_B" + ToString(PARAMS::Beta) + "_X" + ToString(aniso) + "_A"  + ToString(actiontype) + "_"; 
                filename += ToString(step)+"_SM"+ToString(stepstomeasures) + "_PHB" + ToString(nmonte) + "_OR" + ToString(nover) ;

            if(1){        
                vector <int> symmetry={//enum defined in /include/meas/wloopex.h
//                    sigma_g_plus
//                    ,sigma_g_minus
//                    ,sigma_u_plus
//                    ,sigma_u_minus 
//                    ,pi_g
//                    ,pi_u
                    delta_g
//                    ,delta_u
                };
                map <int, int> op_arr{
                    {sigma_g_plus, 9},
                    {sigma_g_minus,12},
                    {sigma_u_plus,12},
                    {sigma_u_minus, 21},
                    {pi_g,12},
                    {pi_u,12},
                    {delta_g,24},
                    {delta_u, 12}
                };
                map <int, int> r_min{
                	{sigma_g_plus, 1},
                	{sigma_g_minus,1},
                	{sigma_u_plus,2},
                	{sigma_u_minus,2},
                	{pi_g, 2}, 
                	{pi_u, 1}, 
                	{delta_g,1}, 
                	{delta_u,2}
                };
                for(int sym_channel:symmetry){
                Timer a0;
                a0.start();
                //Sigma_g_plus<double >  arg(Rmax, Tmax);
                symmetry_sector<double> arg(Rmax, Tmax,op_arr[sym_channel], sym_channel);//Rmax, Tmax, opN, enum symmetry
                cout<<"operator number::" <<arg.opN <<", symmetry::"<<arg.symmetry<<endl;
                ofstream fileout; 
                string filename1 = "/WL_NC"+to_string(NCOLORS)+"_"+get_name_sym(sym_channel) + meastype + "_";
                /*if(meas==1) filename1 = "WilsonLoop_APE_A2_";
                if(meas==2) filename1 = "WilsonLoop_APE_MHIT_A2_";*/
                 filename1 += filename+"_nop_"+ToString(arg.opN)+".dat";
                 string results_dir0="";
//                 string ape_dir="/ape_0.4_"+ToString(meas);
//                 meas==mhit_ape0? ape_dir="/ape0.4_20" :ape_dir="/ape0.5_20";
                 results_dir0=results_dir+"/"+get_name_sym(sym_channel);
                 cmd="mkdir -p "+results_dir0;
                 system(cmd.c_str());
                 filename1=results_dir0+filename1;
                                 

                COUT << "Saving in file: " << filename1 << endl;

                fileout.open(filename1.c_str(), ios::out);
                fileout.precision(16);
                fileout.setf(ios_base::scientific);
                if (!(fileout.is_open())) {
                    COUT << "Error Writing Data File..." << filename1 << endl;    
                    exit(0);
                }
                int numdirs = 3;
                COUT<< "r_min= "<<r_min[sym_channel]<<endl;
                for(int r = r_min[sym_channel]; r <= arg.Rmax; r++){
                    COUT << "###############################################################################" << endl;
                    COUT << "Calculating radius: "<<get_name_sym(sym_channel)<<"::" << r << endl;
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
                            fileout << "\t" << arg.wloop_h[id] ;
                            //if(iop==5) cout << r << "\t" << it  << "\t" << arg.wloop_h[id] << endl; 
                        }
                        fileout << endl;
                    }
                }
                fileout.close();
                a0.stop();
                COUT << "###############################################################################" << endl;
                COUT << "Time for wilson loop calculation: " << a0.getElapsedTime() << " s, for:: " <<get_name_sym(sym_channel) <<endl; 
                COUT << "###############################################################################" << endl;
            }
            }
            BB.Release();
        } // this is end of wilson loop calculation
            // this part is to delete the previous made configuration
            //######################################################
            //######################################################
            if(remove_prv_config && save_config){
                ifstream filein;
                filein.open(fileout_rng_prv.c_str(), ios::binary | ios::in);
            if (filein.is_open() and (iter_prv != termalizationiterterm+stepstomeasures) and (iter_prv!=termalizationiterterm)) {
                bool rng_rm=remove(fileout_prv_config.c_str());
                bool cfg_rm=remove(fileout_rng_prv.c_str());
            if( rng_rm or cfg_rm)
                COUT<< "removing previous config and rng_state failed"<<endl;
            else{
                COUT<<"############################################"<<endl;
                COUT<<fileout_rng_prv<<"\n"<< fileout_prv_config <<"\n"<<" successfully deleted!"<<endl;
                COUT<<"############################################"<<endl;
            }
            }
            filein.close();
            }
            //###########################################################
            //###########################################################
            
            //return;
        }//end of block for accepted configuration
    }//end of loop for maxs
    
    randstates_mhit.Release();
    
    randstates.Release();
    AA.Release();
    COUT << "###############################################################################" << endl;
    COUT << "Time: " << t0.getElapsedTime() << " s" << endl; 
    COUT << "###############################################################################" << endl;
    return ;
}



