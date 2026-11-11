#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <sys/types.h>
#include <stdarg.h>


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
/* HEAT TRANSFER SIMULATION
   
   Simple physical simulation of a rod connected at the left and right
   ends to constant temperature heat/cold sources.  All positions on
   the rod are set to an initial temperature.  Each time step, that
   temperature is altered by computing the difference between a cells
   temperature and its left and right neighbors.  A constant k
   (thermal conductivity) adjusts these differences before altering
   the heat at a cell.  Use the following model to compute the heat
   for a position on the rod according to the finite difference
   method.

      left_diff  = H[t][p] - H[t][p-1];
      right_diff = H[t][p] - H[t][p+1];
      delta = -k*( left_diff + right_diff )
      H[t+1][p] = H[t][p] + delta
   
   Substituting the above, one can get the following

     H[t+1][p] = H[t][p] + k*H[t][p-1] - 2*k*H[t][p] + k*H[t][p+1]

   The matrix H is computed for all time steps and all positions on
   the rod and displayed after running the simulation.  The simulation
   is run for a fixed number of time steps rather than until
   temperatures reach steady state.
*/

int main(int argc, char **argv){
    
    if(argc < 4){
        printf("usage: %s max_time width print\n", argv[0]);
        printf("  max_time: int\n");
        printf("  width: int\n");
        printf("  print: 1 print output, 0 no printing\n");
        return 0;
    }

    int max_time = atoi(argv[1]); // Number of time steps to simulate
    int width = atoi(argv[2]);    // Number of cells in the rod
    int print = atoi(argv[3]);
    double initial_temp = 50.0;   // Initial temp of internal cells 
    double L_bound_temp = 20.0;   // Constant temp at Left end of rod
    double R_bound_temp = 10.0;   // Constant temp at Right end of rod
    double k = 0.5;               // thermal conductivity constant
    double **H;                   // 2D array of temps at times/locations 
    double g = 0.0;
    double *F = &g;               //F is 2D array used to store a row of the final matrix for output


    int num_procs, rank, partner;
    MPI_Init (&argc, &argv);                      // starts MPI 
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);     // get current process id 
    MPI_Comm_size (MPI_COMM_WORLD, &num_procs); // get number of processes
    int n = width/num_procs;
    double send, recv;

    H = malloc(sizeof(double*)*max_time);       //allocate memory for the submatrix on each proc
    int p=0;
    for(int t=0;t<max_time;t++){
        H[t] = malloc(sizeof(double)*n);
    }

    // Initialize temperatures at time 0
    int t = 0;
    for(p=0; p<n; p++){
        H[t][p] = initial_temp;
    }
    // Initialize constant left/right boundary temperatures
    if(rank==0){
        for(t=0; t<max_time; t++){      //set boundary conditions if it is the leftmost submatrix
            H[t][0] = L_bound_temp;
        }
        F = malloc(sizeof(double*)*width);      //also allocate row if it is the root proc since it will handle output
        memset(F,0,sizeof(double*)*width);
    }
    if(rank==num_procs-1){      //set boundary contidions if it is the rightmost submatrix
        for(t=0; t<max_time; t++){
            H[t][n-1] = R_bound_temp;
        }
    }

    
    
    double left_diff, right_diff, delta;

    // Simulate the temperature changes for internal cells
    
    for(t=0; t<max_time-1; t++){            //iterate through each time step
        for(p=1; p<n-1; p++){           //calculate the temperature of non-neighbor values
            left_diff  = H[t][p] - H[t][p-1];
            right_diff = H[t][p] - H[t][p+1];
            delta = -k*( left_diff + right_diff );
            H[t+1][p] = H[t][p] + delta;
        }
        if(num_procs==1){       //don't do anything if only have 1 proc
        }
        else if(rank == 0){      //case for first proc
            partner = 1;
            send = H[t][n-1];           //get data to calculate edge case on the right
            MPI_Sendrecv(&send, 1, MPI_DOUBLE, partner, 1, &recv, 1, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


            p = n-1;
            left_diff  = H[t][p] - H[t][p-1];
            right_diff = H[t][p] - recv;
            delta = -k*( left_diff + right_diff );
            H[t+1][p] = H[t][p] + delta;
        }
        else if(rank == num_procs-1){       //case for last proc
            partner = rank-1;
            send = H[t][0];         //get data to calculate the edge case on the left
            MPI_Sendrecv(&send, 1, MPI_DOUBLE, partner, 1, &recv, 1, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            p = 0;
            left_diff  = H[t][p] - recv;
            right_diff = H[t][p] -  H[t][p+1];
            delta = -k*( left_diff + right_diff );
            H[t+1][p] = H[t][p] + delta;
        }
        else{       //case for all procs in the middle
            if(rank%2==0){      //if rank is even
                send = H[t][n-1];           //first exchange with proc to the right
                partner = rank+1;       
                MPI_Sendrecv(&send, 1, MPI_DOUBLE, partner, 1, &recv, 1, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                p = n-1;
                left_diff  = H[t][p] - H[t][p-1];
                right_diff = H[t][p] - recv;
                delta = -k*( left_diff + right_diff );
                H[t+1][p] = H[t][p] + delta;

                send = H[t][0];         //then exchange with proc to the left
                partner = rank-1;
                MPI_Sendrecv(&send, 1, MPI_DOUBLE, partner, 1, &recv, 1, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                p = 0;
                left_diff  = H[t][p] - recv;
                right_diff = H[t][p] -  H[t][p+1];
                delta = -k*( left_diff + right_diff );
                H[t+1][p] = H[t][p] + delta;
            }
            else{       //if rank is odd
                send = H[t][0];         //first exchange with proc to the left
                partner = rank-1;
                MPI_Sendrecv(&send, 1, MPI_DOUBLE, partner, 1, &recv, 1, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                p = 0;
                left_diff  = H[t][p] - recv;
                right_diff = H[t][p] -  H[t][p+1];
                delta = -k*( left_diff + right_diff );
                H[t+1][p] = H[t][p] + delta;

                send = H[t][n-1];           //then exchange with proc to the right
                partner = rank+1;       
                MPI_Sendrecv(&send, 1, MPI_DOUBLE, partner, 1, &recv, 1, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                

                p = n-1;
                left_diff  = H[t][p] - H[t][p-1];
                right_diff = H[t][p] - recv;
                delta = -k*( left_diff + right_diff );
                H[t+1][p] = H[t][p] + delta;
            }
        }
    }
    
    if(print){      // output the final matrix if specified by user
        if(rank==0){
            printf("%3s| ","");
            for(p=0; p<width; p++){
                printf("%5d ",p);
            }
            printf("\n");
            printf("%3s+-","---");
            for(p=0; p<width; p++){
                printf("------");
            }
            printf("\n");
        }
        // Row headers and data
        for(t=0; t<max_time; t++){
            if(MPI_Gather(H[t],n,MPI_DOUBLE,F,n,MPI_DOUBLE,0,MPI_COMM_WORLD)){      //get the whole row of that timestep from all subrows accross all procs into F for printing purposes
                printf("gather error\n");
            }
            if(rank==0){        // only print if it is the root proc
                printf("%3d| ",t);
                for(p=0; p<width; p++){
                    printf("%5.1f ",F[p]);
                }
                printf("\n");
            }
            
        }
    }
    
    for(t=0; t<max_time; t++){          //free allocated memory from above
        free(H[t]);
    }
    free(H);
    dpprintf("here\n");
    if(rank==0){
        free(F);
    }
    MPI_Finalize();
    return 0;
}
      
