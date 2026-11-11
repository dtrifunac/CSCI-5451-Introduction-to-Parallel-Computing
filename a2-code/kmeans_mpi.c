#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <mpi.h>
#include <stdarg.h>

// #include "filestats.c"

typedef struct KMData KMData;
typedef struct KMClust KMClust;

struct KMData {
    int ndata;
    int dim;
    double **features;
    int *assigns;
    int *labels;
    int nlabels;
};

struct KMClust {
    int nclust;
    int dim;
    double **features;
    int *counts;
};


int filestat(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines){
// Sets number of lines and total number of whitespace separated
// tokens in the file. Returns -1 if file can't be opened, 0 on
// success.
//
// EXAMPLE: int ret = filestats("digits_all_1e1.txt", &toks, &lines);
// toks  is now 7860 : 10 lines with 786 tokens per line, label + ":" + 28x28 pixels
// lines is now 10   : there are 10 lines in the file
  FILE *fin = fopen(filename,"r");
  if(fin == NULL){
    printf("Failed to open file '%s'\n",filename);
    return -1;
  }

  ssize_t ntokens=0, nlines=0, column=0;
  int intoken=0, token;
  while((token = fgetc(fin)) != EOF){
    if(token == '\n'){          // reached end of line
      column = 0;
      nlines++;
    }
    else{
      column++;
    }
    if(isspace(token) && intoken==1){ // token to whitespace
      intoken = 0;
    }
    else if(!isspace(token) && intoken==0){ // whitespace to token
      intoken = 1;
      ntokens++;
    }
  }
  if(column != 0){              // didn't end with a newline
    nlines++;                   // add a line on to the count
  }
  *tot_tokens = ntokens;
  *tot_lines = nlines;
  fclose(fin);
  // printf("DBG: tokens: %lu\n",ntokens);
  // printf("DBG: lines: %lu\n",nlines);
  return 0;
}

KMData *kmdata_load(char *datafile, double *all) {
    ssize_t toks;
    ssize_t lines;
    filestat(datafile,&toks,&lines);
    KMData *data = (KMData *)malloc(sizeof(KMData));

    data->ndata = (int)lines;
    int numtoks = (int)toks/(int)lines;
    data->dim = numtoks-2;
    // printf("dim: %d\n", data->dim);
    // printf("ndatata: %d\n", data->ndata);
    data->features = (double**)malloc(data->ndata * sizeof(double *));
    for (int i = 0; i < data->ndata; i++) {
        data->features[i] = &all[i*data->dim];
    }

    data->assigns = malloc(data->ndata * sizeof(int));
    data->labels = malloc(data->ndata * sizeof(int));
    data->nlabels = 0;
    // int i = 0;
    // char *token;
    int maxlab = 0;
    FILE *fin = fopen(datafile, "r");
    char line[10000];
    for(int i = 0; i < data->ndata; i++) {
        char* r = fgets(line, 10000, fin);
        if(r==NULL){
            printf("ERROR FGETS\n\n\n\n\n");
        }
        char* token = strtok(line, " ");
        if(token == NULL){
            printf("token is null\n");
        }
        data->labels[i] = atoi(token);
        if(data->labels[i] > maxlab){
            maxlab = data->labels[i];
        }
        token = strtok(NULL, "   ");
        for(int j = 0; j < data->dim; j++) {
            token = strtok(NULL, "   ");
                data->features[i][j] = atof(token);
        }
    }

    data->nlabels = maxlab+1;
    fclose(fin);
    return data;
}

KMClust *kmclust_new(int nclust, int dim, double* all) {
    KMClust *clust = (KMClust *)malloc(sizeof(KMClust));
    clust->nclust = nclust;
    clust->dim = dim;
    
    clust->features = malloc(nclust * sizeof(double *));
    for (int i = 0; i < nclust; i++) {
        clust->features[i] = &all[i*dim];
        for (int j = 0; j < dim; j++) {
            clust->features[i][j] = 0.0;
        }
    }
    clust->counts = malloc(nclust * sizeof(int));
    for (int i = 0; i < nclust; i++) {
        clust->counts[i] = 0;
    }
    return clust;
}

void save_pgm_files(KMClust *clust, char *savedir) {
    int dim_root =  (int)sqrt(clust->dim);
    if (clust->dim % dim_root == 0) {
        printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);
        double maxfeat = 0.0;
        for (int i = 0; i < clust->nclust; i++) {
            for (int j = 0; j < clust->dim; j++) {
                if (clust->features[i][j] > maxfeat) {
                    maxfeat = clust->features[i][j];
                }
            }
        }
        for(int c =0; c < clust->nclust; c++){
            char outfile[256];
            sprintf(outfile, "%s/cent_%04d.pgm", savedir, c);
            FILE* pgm = fopen(outfile, "w");
            fprintf(pgm, "P2\n");
            fprintf(pgm, "%2d  %2d\n", dim_root, dim_root);
            fprintf(pgm, "%3.0f\n", maxfeat);
            for(int d =0; d < clust->dim; d++){
                if(d>0 && d%dim_root == 0){
                    fprintf(pgm, "\n");
                }
                fprintf(pgm, "%3.0f ", clust->features[c][d]);
            }
            fprintf(pgm, "\n");
            fclose(pgm);
        }
    }
}





int main(int argc, char **argv){
    int num_procs, rank, n, dim, surplus, els_per_proc, num_labels;
    MPI_Init(&argc, &argv);                      // starts MPI 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // get current process id 
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // get number of processes
    if(argc<3){
        printf("error");
        return -1;
    }
    char* datafile = (char*)argv[1];
    int nclust = atoi(argv[2]);
    int MAXITER =100;
    char *savedir = ".";
    if(rank==0){        //root proc handles output
        if(argc>3){
            savedir = argv[3];
            mkdir(savedir,0777);
        }
        
        
    }
    if(argc>4){
        MAXITER =atoi(argv[4]);
    }
    double *all;        //used to store all the data features
    KMData *masdata;        //used as the struct to get information about all the data
    if(rank==0){ 
        printf("datafile: %s\n", datafile);
        printf("nclust: %d\n", nclust);
        printf("savedir: %s\n", savedir);


        ssize_t toks;
        ssize_t lines;
        filestats(datafile,&toks,&lines);
        all = malloc(sizeof(double)*(int)lines*((int)toks-2));
        masdata = kmdata_load(datafile, all);
        printf("ndata: %d\n",masdata->ndata);
        printf("dim: %d\n\n",masdata->dim);
        n = masdata->ndata/num_procs;
        surplus = masdata->ndata%num_procs;
        dim = masdata->dim;
        num_labels = masdata->nlabels;
        
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);       //get info relating to problem size to all procs
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&surplus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    KMData *data = (KMData *)malloc(sizeof(KMData));        //used to store the local data assigned to each proc

   
    els_per_proc = n*dim;
    int *counts = malloc(num_procs*sizeof(int));        //used for the vector MPI operations, counts is number of data elements
    int *numrows = malloc(num_procs*sizeof(int));       //numrows is number of rows of data each proc will get
    int *displs = malloc(num_procs*sizeof(int));        //displacements for counts
    int *displs2 = malloc(num_procs*sizeof(int));       //displacements for numrows
    for(int i = 0; i<num_procs; i++){                   //set these variables depending on which proc it is
        counts[i] = (i<surplus) ? els_per_proc+dim : els_per_proc;
        numrows[i] = (i<surplus) ? n+1 : n;
        displs[i] = (i==0) ? 0 : displs[i-1] + counts[i-1];
        displs2[i] = (i==0) ? 0 : displs2[i-1] + numrows[i-1];
    }
    data->features = (double**)malloc(n * sizeof(double *));        //initiliaze the fields of data
    double* myrows = malloc(sizeof(double) * counts[rank]);         //used to store the feautures of local data
    for (int i = 0; i < numrows[rank]; i++) {
        data->features[i] = &myrows[i*dim];
    }
    data->assigns = malloc(numrows[rank] * sizeof(int));
    data->labels = malloc(numrows[rank] * sizeof(int));
    data->ndata = numrows[rank];
    data->dim = dim;

    
    MPI_Scatterv(all, counts, displs, MPI_DOUBLE, myrows, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);     //scatter the whole data features to local data structs
    data->nlabels = num_labels;

    double* myclusts = malloc(sizeof(double)*dim*nclust);       //used for vector ops on clust's features
    KMClust *clust = kmclust_new(nclust, data->dim, myclusts);      //personal clust stored in memory on each proc
    
    int c;
    for(int c = 0; c < clust->nclust; c++){     //set counts of all clusts to zero
        clust->counts[c] = 0;
    } 
    for(int i = 0; i < data->ndata; i++){       //set the assignment of each data and augment the counts of the respective cluster
        c = (i+displs2[rank])%clust->nclust;

        data->assigns[i] = c;
        clust->counts[c]++;
        
    }


    MPI_Allreduce(MPI_IN_PLACE, clust->counts, nclust, MPI_INT, MPI_SUM, MPI_COMM_WORLD);       //add up the counts of each cluster on all procs so they all have the total among all procs


    int curiter = 1;                       
    int nchanges = data->ndata;    
    if(rank==0){           
        printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
        printf("ITER NCHANGE CLUST_COUNTS\n");
    }
    while(nchanges > 0 && curiter <= MAXITER){          //iterate until MAXITER is met or data have converged
        for(int c = 0; c < clust->nclust; c++){     //initialize cluster feature centers for easy calculation in a bit
            for(int d = 0; d < clust->dim; d++){
                clust->features[c][d] = 0.0;
            }
        }
        
        for(int i = 0; i < data->ndata; i++){       //augment feautures of each cluster by the data in that cluster
            int c = data->assigns[i];
            for(int d = 0; d < clust->dim; d++){
                clust->features[c][d] += data->features[i][d];
            }
        }
    
        MPI_Allreduce(MPI_IN_PLACE, myclusts , nclust*dim , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);     // get the global total of each feature of each cluster to all procs

        for(int c = 0; c < clust->nclust; c++){     //calculate the actual value of the features of each cluster by taking the average over the number of members of that cluster
            if(clust->counts[c]>0){
                for(int d = 0;d<clust->dim;d++){
                    clust->features[c][d] = clust->features[c][d]/clust->counts[c];
                    if(rank==0){
                        // printf("c: %d d: %d clust features: %3f\n",c,d,clust->features[c][d]);
                    }
                }
            }
        }
        for(int c=0;c<clust->nclust;c++){       // set the counts of each cluster back to zero to make it easy to see number of members after calculating assignments
            clust->counts[c]=0;
        }

        nchanges = 0;
        for(int i = 0; i<data->ndata;i++){      //double check cluster assignments of each data now that the features of each cluster is updated
            int best_clust = 0;
            float best_distsq = INFINITY;
            for(int c=0;c<clust->nclust;c++){
                float distsq =0.0;
                for(int d=0;d<clust->dim;d++){      //calculate the total distance from that cluster
                    float diff= data->features[i][d] - clust->features[c][d];
                    distsq += diff*diff;
                    // printf("DBG: curiter %d i %d c %d d %d diff %.6f\n",curiter,i,c,d,diff);

                }
                if(distsq < best_distsq){       //if it is the smallest distance this is the best cluster assignment
                    best_clust = c;
                    best_distsq = distsq;
                }
            }
            clust->counts[best_clust] +=1;
            if(best_clust != data->assigns[i]){     //reassign to the best cluster assinment if necessary
                // printf("DBG: data %d changing clusters: %d to %d\n",i+displs2[rank],data->assigns[i],best_clust);
                nchanges++;
                data->assigns[i] = best_clust;
            }
        }

        
        MPI_Allreduce(MPI_IN_PLACE, &nchanges , 1 , MPI_INT , MPI_SUM , MPI_COMM_WORLD);        //communicate global number of changes to all procs
        MPI_Allreduce(MPI_IN_PLACE, clust->counts, nclust, MPI_INT, MPI_SUM, MPI_COMM_WORLD);       //communicate number of data in each cluster to all procs

        if(rank==0){        //print out the number of members of each proc from the root proc
            printf("%4d:  %4d | ",curiter,nchanges);
            for(int c=0;c<nclust;c++){
                printf("%4d  ",clust->counts[c]);
            }
            printf("\n");
        }
        curiter++;
    }
    FILE* fout;
    int *recv;
    if(rank==0){        //allocate memory to sore the final label for all data on the root proc
        recv = malloc(masdata->ndata*sizeof(int));
    }
    MPI_Gatherv(data->assigns, numrows[rank], MPI_INT,recv, numrows, displs2, MPI_INT, 0, MPI_COMM_WORLD);      //send the label of all data to the root proc
    if(rank==0){
        if(curiter > MAXITER){
            printf("WARNING: maximum iteration %d exceeded, may not have conveged\n",MAXITER);
        }
        else{
            printf("CONVERGED: after %d iterations\n",curiter);
        }
        printf("\n");
        printf("==CONFUSION MATRIX + COUNTS==\n");
        printf("LABEL \\ CLUST\n");
        printf("\n");
        int **confusion = (int **)malloc(masdata->nlabels * sizeof(int *));             // allocate memory for the confusion matrix
        for (int i = 0; i < masdata->nlabels; i++) {
            confusion[i] = (int *)malloc(clust->nclust * sizeof(int));
            for (int j = 0; j < clust->nclust; j++) {
                confusion[i][j] = 0;
            }
        }
        
        for(int i = 0;i<masdata->ndata;i++){        //set the fields of the confusion matrix
            confusion[masdata->labels[i]][recv[i]]++;
        }

        printf("   ");    
        for(int j = 0; j < clust->nclust; j++){     //print the header
            printf(" %4d",j);
        }
        printf("  TOT\n");
        int tot = 0;
        for(int i=0; i < masdata->nlabels; i++){            //print out the confusion matrix
            printf(" %d:",i);
            tot = 0;
            for(int j=0; j < clust->nclust; j++){
                printf(" %4d",confusion[i][j]);
                tot = tot + confusion[i][j];
            }
            printf(" %3d\n",tot);
        }

        printf("TOT");
        tot = 0;
        for(int c = 0; c < clust->nclust; c++){     //print out the totals
            printf("%5d",clust->counts[c]);
            tot = tot + clust->counts[c];
        }
        printf(" %4d\n\n",tot);
        
        char outfile[256];
        sprintf(outfile, "%s/labels.txt", savedir);
        printf("Saving cluster labels to file %s\n", outfile);
        fout = fopen(outfile, "w");
        for(int i = 0; i < masdata->ndata; i++){
            fprintf(fout, "%3d %3d\n", masdata->labels[i], recv[i]);
        }
        save_pgm_files(clust, savedir);             //save to the specified output file
        for (int i = 0; i < data->nlabels; i++) {       //free memory for the confusion matrix
            free(confusion[i]);
        }
        free(confusion);
        fclose(fout);
    }
    

    free(myrows);               //free all memory allocated at the beginning
    free(data->features);

    free(data->assigns);
    free(data->labels);
    free(data);


    if(rank==0){
        free(all);
        free(recv);
        free(masdata->features);

        free(masdata->assigns);
        free(masdata->labels);
        free(masdata);
        
    }


    free(myclusts);
    free(clust->features);
    free(clust->counts);

    free(clust);

    free(counts);
    free(displs);

    free(numrows);
    free(displs2);

    
    
    MPI_Finalize();
}