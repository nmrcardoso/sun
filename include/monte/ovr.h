
#ifndef OVR_H
#define OVR_H


#include <typeinfo>
#include <gaugearray.h>
#include <random.h>

#include <timer.h>
#include <tune.h>

namespace CULQCD{





template <class Real> 
class OverRelaxation: Tunable{
private:
   typedef void (*TFuncPtr)(complex*, int, int);
   TFuncPtr kernel_pointer;
   gauge array;
   int size;
   double timesec;
   int grid[4];
   bool tex;
#ifdef TIMMINGS
    Timer mtime;
#endif
    int dir, parity;

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream);
   void SetFunctionPtr();

public:
   OverRelaxation(gauge &array);
   ~OverRelaxation(){};


   void Run(const cudaStream_t &stream);
   void Run();
   double flops();
   double bandwidth();
   long long flop() const ;
   long long bytes() const;
   double time();
   void stat();
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << grid[0] << "x";
    vol << grid[1] << "x";
    vol << grid[2] << "x";
    vol << grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), array.ToStringArrayType().c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { array.Backup(); }
  void postTune() { array.Restore(); }
};






















}

#endif

