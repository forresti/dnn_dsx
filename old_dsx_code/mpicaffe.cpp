#include <glog/logging.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
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
aprun -n 4 -d 16 ./build/tools/mpicaffe
aprun -n 4 -d 16 $CAFFE_ROOT/build/tools/mpicaffe
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

// Train / Finetune a model.
//@param solver_path = /path/to/train_dir/solver.prototxt
//@param solverstate = /path/to/train_dir/caffe_train_iter_90000.solverstate
int train(string solver_path, string solverstate) {
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(solver_path, &solver_param);

  Caffe::SetDevice(0);
  Caffe::set_mode(Caffe::GPU);

  LOG(INFO) << "Starting Optimization";

  shared_ptr<caffe::Solver<float> > solver(caffe::GetSolver<float>(solver_param)); //original. HANGS HERE when using MPI 
  LOG(INFO) << "Initialized solver";
  google::FlushLogFiles(google::GLOG_INFO); // good to flush this sometimes... 

  if(!solverstate.empty()){
    LOG(INFO) << "Resuming from " << solverstate;
    solver->Solve(solverstate);
  }
  else{
    solver->Solve();
  }

  LOG(INFO) << "Optimization Done.";
  return 0;
}

//@param train_list_file = file containing list of dirs to train in
//       e.g. "./nets/0 \n ./nets/1 ... etc"
//@return train_dirs = list of directories to train in (one per line in train_list_file)
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

//@param train_dir = directory to look for *.solverstate snapshot
//                  (prefer absolute path. relative path will work if we're in the right directory, though.)
//@return latest snapshot (e.g. caffe_train_iter_95000.solverstate)
string get_latest_solverstate(string train_dir){

  //get list of solverstate files in train_dir
  DIR *dir = opendir(train_dir.c_str());
  if(!dir){
    LOG(ERROR) << "failed to open dir: " << train_dir;
  }
  vector<string> solverstates;
  struct dirent *ent = readdir(dir);
  while(ent){ 
    string fname = string(ent->d_name);
    if( fname.find(".solverstate") == string::npos ){ 
      //this isn't a *.solverstate string
    }
    else{
      fname = train_dir + "/" + fname;
      solverstates.push_back(fname);
    }
    ent = readdir(dir);
  }

  time_t newest_file_time = 0;
  string newest_file_name = "";

  //find newest solverstate file
  for(int i=0; i<solverstates.size(); i++){
    struct stat stat_buf; //in sys/stat.h
    string fname = solverstates[i];
    stat(fname.c_str(), &stat_buf);
    time_t last_modified = stat_buf.st_mtime;
    //LOG(ERROR) << fname << " last modified: " << last_modified; 

    if(last_modified > newest_file_time){
      //found a newer solverstate file
      newest_file_time = last_modified;
      newest_file_name = fname;
    }
  }
  //LOG(ERROR) << "newest solverstate: " << newest_file_name;
  return newest_file_name;
}

//thanks: stackoverflow.com/questions/12774207
inline bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}


void my_init_logging(string train_dir){
  //thx: http://www.cplusplus.com/reference/ctime/strftime 
  char now[200];
  time_t rawtime;
  struct tm * timeinfo;

  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(now, 200, "%a_%Y_%m_%d__%H_%M_%S", timeinfo);
  string log_fname = train_dir + "/train_" + string(now) + ".log";
  LOG(ERROR) << "log_fname: " << log_fname;

  //thx: https://code.google.com/p/google-glog/issues/detail?id=26
  google::SetLogDestination(google::INFO, log_fname.c_str()); //includes INFO, WARNING, and ERROR.
  //google::SetLogFilenameExtension(""); //default: append datetime. but, we already include datetime in our log filename.
  //google::SetLogFilenameExtension(".log"); //is ignored... still names the logs "...log.datetime"
}

int main(int argc, char** argv) {

  int rank, nproc;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  printf("  Rank: %d Total: %d \n",rank, nproc);

  //TODO: take cmd-line args... eventually.
  string dnn_dsx_dir("/lustre/atlas/scratch/forresti/csc103/dnn_exploration/dnn_dsx/");
  string train_list = dnn_dsx_dir + "/" +  "train_list.txt";

  vector<string> train_dirs = get_train_dirs(train_list);
  if(rank < train_dirs.size()){
    string my_train_dir = dnn_dsx_dir + "/" + train_dirs[rank];
    string solverstate = get_latest_solverstate(my_train_dir);
    LOG(ERROR) << "rank:" << rank << ", my_train_dir:" << my_train_dir << ", solverstate:" << solverstate;

    //assumption -- the Caffe solver is located here: my_train_dir/solver.prototxt
    string solver_path = my_train_dir + "/solver.prototxt";
    if( file_exists(solver_path) ){

      my_init_logging(my_train_dir);
      FLAGS_logbufsecs = 0;  //flush logs often (only works if placed BEFORE InitGoogleLogging)
      caffe::GlobalInit(&argc, &argv);  //calls InitGoogleLogging()

      train(solver_path, solverstate);
    }
    else{
      LOG(ERROR) << "rank:" << rank << ", solver not found: " << solver_path;
    }

  }
  //else, this rank does no work.
  MPI_Finalize();

}
