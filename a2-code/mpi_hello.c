// Slightly more advanced hello world. Each processor reports its
// proc_id and the host machine on which it is running. Includes a
// debug utility for printing messages prepended with the proc_id and
// printing only on the root.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <sys/types.h>

// Prints out a message if the environment variable DEBUG is set;
// includes processor information.
// Try running as `DEBUG=1 mpirun -np 4 ./mpi_hello`
void dpprintf(const char* format, ...) { 
  if(getenv("DEBUG") != NULL){
    int total_procs, proc_id, host_len;
    char host[256];
    MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     // get current process id 
    MPI_Comm_size (MPI_COMM_WORLD, &total_procs); // get number of processes 
    MPI_Get_processor_name(host, &host_len);      // get the symbolic host name 
    pid_t pid = getpid();

    va_list args;
    va_start (args, format);
    char fmt_buf[2048];
    snprintf(fmt_buf, 2048, "|DEBUG Proc %03d / %d PID %d Host %s| %s",
             proc_id, total_procs, pid, host, format);
    vfprintf(stderr, fmt_buf, args);
    va_end(args);
  }
}  

// Initialize MPI and relevant variables then have each proc say hello  
int main (int argc, char *argv[]){
  int total_procs, proc_id, host_len;
  char host[256];

  MPI_Init (&argc, &argv);                      // starts MPI 
  MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     // get current process id 
  MPI_Comm_size (MPI_COMM_WORLD, &total_procs); // get number of processes 
  MPI_Get_processor_name(host, &host_len);      // get the symbolic host name 

  // Each processor prints; message is appended with proc # 
  printf("Hello world from process %d of %d (host: %s)\n",
         proc_id, total_procs, host);

  // If run with DEBUG=1, each proc prints a debug message
  dpprintf("Debug message from processor %d\n",proc_id);

  // Only the root processor 0 prints 
  if(proc_id == 0){
    printf("Hello from the root processor %d of %d (host: %s)\n",
           proc_id, total_procs, host);
  }
  MPI_Finalize();
  return 0;
}
