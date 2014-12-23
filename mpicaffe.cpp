#include <glog/logging.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <dirent.h>
#include "mpi.h"
#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
//using caffe::vector; //what is this? -Forrest
using std::vector;
using std::string;
using std::ifstream;

/*
running this code:
aprun -n 4 -d 1 ./build/tools/mpicaffe
    #TODO: flags.

*/

DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  LOG(INFO) << "Starting Optimization";
  shared_ptr<caffe::Solver<float> > solver(caffe::GetSolver<float>(solver_param));

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Solve(FLAGS_snapshot);
  } 
 else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);

//@param train_list_file = file containing list of dirs to train in
//       e.g. "./nets/0 \n ./nets/1 ... etc"
vector<string> get_train_dirs(string train_list_file){
  vector<string> train_dirs;

  ifstream infile(train_list_file.c_str());  
  if(!infile.is_open()){
    LOG(ERROR) << "cannot open train_list: " << train_list_file;
  }
  string tmp_str;
  while(infile >> tmp_str){
    train_dirs.push_back(tmp_str);
  } 
     
  return train_dirs;
}

int main(int argc, char** argv) {

  int rank,nproc;
  int i;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  printf("  Rank: %d Total: %d \n",rank, nproc);

  string train_list("/lustre/atlas/scratch/forresti/csc103/dnn_exploration/dnn_dsx/train_list.txt"); //TODO: take as input arg.
  //TODO: parse args here?

  vector<string> train_dirs = get_train_dirs(train_list);
  if(rank < train_dirs.size()){
    string my_train_dir = train_dirs[rank];
    printf("  Rank: %d  my_train_dir: %s \n",rank, my_train_dir.c_str());

  }
  //else, this rank does no work.
  MPI_Finalize();

  #if 0
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
  #endif
}
