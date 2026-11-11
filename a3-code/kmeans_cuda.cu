#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#define CHECK(EXPR) (assert(cudaSuccess == (EXPR)))
// #include "filestats.c"

typedef struct KMData KMData;
typedef struct KMClust KMClust;

struct KMData {
    int ndata;
    int dim;
    float **features;
    float *assigns;
    int *labels;
    int nlabels;
};

struct KMClust {
    int nclust;
    int dim;
    float **features;
    float *counts;
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
  return 0;
}


// load in data to a KMData struct
KMData *kmdata_load(char *datafile) {
    ssize_t toks;
    ssize_t lines;
    filestat(datafile,&toks,&lines);
    KMData *data = (KMData *)malloc(sizeof(KMData));

    data->ndata = (int)lines;
    int numtoks = (int)toks/(int)lines;
    data->dim = numtoks-2;

    data->features = (float**)malloc(data->ndata * sizeof(float *));
    for (int i = 0; i < data->ndata; i++) {
        data->features[i] = (float*)malloc(data->dim * sizeof(float));
    }

    data->assigns = (float*)malloc(data->ndata * sizeof(float));
    data->labels = (int*)malloc(data->ndata * sizeof(int));
    data->nlabels = 0;
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

// initialize a new km clust struct
KMClust *kmclust_new(int nclust, int dim) {
    KMClust *clust = (KMClust *)malloc(sizeof(KMClust));
    clust->nclust = nclust;
    clust->dim = dim;
    clust->features = (float**)malloc(nclust * sizeof(float *));
    for (int i = 0; i < nclust; i++) {
        clust->features[i] = (float*)malloc(dim * sizeof(float));
        memset(clust->features[i],0,sizeof(float)*dim);
    }
    clust->counts = (float*)malloc(nclust * sizeof(float));
    for (int i = 0; i < nclust; i++) {
        clust->counts[i] = 0;
    }
    return clust;
}


// save clust info to a pgm file
void save_pgm_files(KMClust *clust, char *savedir) {
    int dim_root =  (int)sqrt(clust->dim);
    if (clust->dim % dim_root == 0) {
        printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);
        float maxfeat = 0.0;
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



// used to calculate the new cluster centers in the GPU.
// each block is assigned to one cluster. each thread is assigned to one dimension of that cluster
__global__ void calc_clust_centers( float *dev_data_all_feats, 
                                    float *dev_clust_all_feats, 
                                    float* dev_data_assigns, int ndata, int dim)
{
    // thread's index and the block's index used for determining cluster assignment
    long idx = threadIdx.x;
    long blockidx = blockIdx.x;
    
    // iterate through all the data
    for(int i = 0; i < ndata; i++){
        int c = (int)dev_data_assigns[i];
        
        // if block is assigned to the cluster this datapoint belongs to
        if(c==blockidx){
            // then add its feature in that dimention to the cluster's features
            dev_clust_all_feats[c*dim+idx]+=dev_data_all_feats[i*dim+idx];
        }
    }
}



// function used to divide cluster centers by the number of data in them to get the average.
// each block is assigned to one cluster. Each thread within a block is assigned one dimension
__global__ void div_clust_centers(float* dev_clust_all_feats, float* dev_clust_counts, int dim)
{
    long idx = threadIdx.x;
    long blockidx = blockIdx.x;
    int c = (int)blockidx;
    if(dev_clust_counts[c]>0){
        dev_clust_all_feats[c*dim+idx] = dev_clust_all_feats[c*dim+idx]/(int)dev_clust_counts[c];
    }
}


// function to make the GPU print the number of data in each cluster. do this to avoid unnecessary communication with CPU
__global__ void print_clust_counts(float* dev_clust_counts, int nclust)
{
    // loop through each cluster and print its count
    for(int c=0;c<nclust;c++){
        printf("%4d  ",(int)dev_clust_counts[c]);
    }
    printf("\n");
}


// function to make GPU assign data points to clusters
__global__ void assign_clusts(   float *dev_data_all_feats, 
                            float *dev_clust_all_feats, 
                            float* dev_data_assigns, 
                            float* dev_clust_counts, 
                            int nclust, int dim, float* nchange)
{
    // each thread is assigned to 1 datapoint
    long idx = threadIdx.x + blockDim.x*blockIdx.x;
    int i = (int)idx;

    // start algorithm
    int best_clust = 0;
    float best_distsq = INFINITY;
    // for each cluster
    for(int c=0;c<nclust;c++){
        float distsq =0.0;
        // calculate the distance metric based on the data's features
        for(int d=0;d<dim;d++){
            float diff = dev_data_all_feats[i*dim+d] - dev_clust_all_feats[c*dim+d];
            distsq += diff*diff;
        }
        // check if this cluster is a better assignment
        if(distsq < best_distsq){
            best_clust = c;
            best_distsq = distsq;
        }
    }
    // atomically augment the count of the cluster it got assigned to
    atomicAdd(&dev_clust_counts[best_clust],1);
    if(best_clust != dev_data_assigns[i]){ 
        // atomically augment the number of changes if there was a change in assignment
        atomicAdd(nchange,1);
        // and reasign the datapoint to its new cluster
        dev_data_assigns[i] = best_clust;
    }
}



int main(int argc, char **argv){
    if(argc<3){
        printf("error");
        return -1;
    }

    // parse the user inputs
    char* datafile = (char*)argv[1];
    int nclust = atoi(argv[2]);
    int MAXITER =100;
    char *savedir = ".";
    
    // set variables accordingly
    if(argc>3){
        savedir = argv[3];
        mkdir(savedir,0777);
    }
    if(argc>4){
        MAXITER =atoi(argv[4]);
    }
    printf("datafile: %s\n", datafile);
    printf("nclust: %d\n", nclust);
    printf("savedir: %s\n", savedir);

    // load in data and cluster struct
    KMData *data = kmdata_load(datafile);
    KMClust *clust = kmclust_new(nclust, data->dim);

    printf("ndata: %d\n",data->ndata);
    printf("dim: %d\n\n",data->dim);


    // make initialze cluster assignments
    int c;
    for(int i = 0; i < data->ndata; i++){
        c = i%clust->nclust;
        data->assigns[i] = c;
    }

    // set the counts of each cluster accordingly
    for(int c = 0; c < clust->nclust; c++){
        int icount = round(data->ndata/clust->nclust);
        int extra = 0;
        if(c < (data->ndata % clust->nclust)){
            extra = 1;
        }
        clust->counts[c] = icount + extra;
    }

    // initialize variables and memory to be used in the GPU
    float *dev_data_all_feats, *dev_clust_all_feats, *dev_data_assigns, *dev_clust_counts, *nchange;
    int ndata = data->ndata;
    int dim = clust->dim;
    CHECK(cudaMalloc((void**)&dev_data_all_feats, dim*ndata*sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_clust_all_feats, nclust*dim*sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_data_assigns, ndata*sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_clust_counts, nclust*sizeof(float)));
    CHECK(cudaMalloc((void**)&nchange, sizeof(float)));
    CHECK(cudaMemcpy(dev_data_assigns, data->assigns, ndata*sizeof(float), cudaMemcpyHostToDevice));


    // copy in the data's features to the allocated memory in GPU
    for(int i = 0; i < data->ndata; i++){
        CHECK(cudaMemcpy(dev_data_all_feats+i*dim, data->features[i], dim*sizeof(float), cudaMemcpyHostToDevice));
    }

    // copy data assignmnents into GPU
    CHECK(cudaMemcpy(dev_data_assigns, data->assigns, ndata*sizeof(float), cudaMemcpyHostToDevice));
    // copy cluster counts into GPU memory
    CHECK(cudaMemcpy(dev_clust_counts, clust->counts, nclust * sizeof(float), cudaMemcpyHostToDevice));

    // initialize number of blocks and threads used based on problem size
    int nblocks;
    int nthreads;
    if(data->ndata>=1000){
        nblocks = data->ndata/1000;
        nthreads = 1000;
    }
    else{
        nblocks = 1;
        nthreads = data->ndata;
    }
    

    int curiter = 1;                       
    int nchanges = data->ndata;               
    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");

    // float max = (float)MAXITER;
    // CHECK(cudaMemcpy(maxit, &max, sizeof(float), cudaMemcpyHostToDevice));

    // run_calculation<<<1,1>>>(   dev_data_all_feats,dev_clust_all_feats, 
                                // dev_data_assigns, dev_clust_counts, nchange, 
                                // maxit, ndata, dim, nclust, nblocks, nthreads);

    // iterate until reached convergence
    while(nchanges > 0 && curiter <= MAXITER){
        // make sure to zero out the cluster features so we only calculate based on valid data assignments
        CHECK(cudaMemset(dev_clust_all_feats, 0,nclust*dim*sizeof(float)));
        // calculate the cluster centers before dividing
        calc_clust_centers<<<nclust,dim>>>(dev_data_all_feats, dev_clust_all_feats, dev_data_assigns, ndata, dim);
   
        // divide each cluster center by the number of data points in that cluster
        div_clust_centers<<<nclust,dim>>>(dev_clust_all_feats, dev_clust_counts, dim);

        // set counts back to zero to see how many new assignments there are
        CHECK(cudaMemset(dev_clust_counts, 0, nclust*sizeof(float)));
        // reset the number of changes to 0 to see if any changes occured
        CHECK(cudaMemset(nchange, 0,sizeof(float)));

        // calculate cluster assignments
        assign_clusts<<<nblocks,nthreads>>>(dev_data_all_feats, dev_clust_all_feats, dev_data_assigns, dev_clust_counts, nclust, dim, nchange);

        // export number of changes from device to host for printing and the outer loop
        float ch = 0;
        CHECK(cudaMemcpy(&ch, nchange,   sizeof(float), cudaMemcpyDeviceToHost));
        nchanges = (int)ch;
        printf("%4d:  %4d | ",curiter,nchanges);

        // print the number of data in each cluster. Do this is the gpu since all the data is already loaded there
        print_clust_counts<<<1,1>>>(dev_clust_counts, nclust);

        curiter++;
    }

    // copy memory back into the CPU for confusion matrix printing
    CHECK(cudaMemcpy(data->assigns,dev_data_assigns,  ndata * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(data->assigns,dev_data_assigns,  ndata * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(clust->counts,dev_clust_counts,  nclust * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nclust; i++) {
            CHECK(cudaMemcpy(clust->features[i], dev_clust_all_feats+i*dim, dim*sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaDeviceSynchronize();

    // begin printing information in the CPU
    if(curiter > MAXITER){
        printf("WARNING: maximum iteration %d exceeded, may not have conveged\n",MAXITER);
    }
    else{
        printf("CONVERGED: after %d iterations\n",curiter);
    }
    printf("\n");
    

    int **confusion = (int **)malloc(data->nlabels * sizeof(int *));
    for (int i = 0; i < data->nlabels; i++) {
        confusion[i] = (int *)malloc(clust->nclust * sizeof(int));
        memset(confusion[i],0,sizeof(int)*clust->nclust);
    }

    
    for(int i = 0;i<data->ndata;i++){
        confusion[data->labels[i]][(int)data->assigns[i]]++;
    }
    printf("==CONFUSION MATRIX + COUNTS==\n");
    printf("LABEL \\ CLUST\n");
    printf("\n");

    printf("   ");    
    for(int j = 0; j < clust->nclust; j++){
        printf(" %4d",j);
    }
    printf("  TOT\n");
    int tot = 0;
    for(int i=0; i < data->nlabels; i++){
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
    

    for(int c = 0; c < clust->nclust; c++){
        printf("%5d",(int)clust->counts[c]);
        tot = tot + clust->counts[c];
    }
    printf(" %4d\n\n",tot);

    
    char outfile[256];
    sprintf(outfile, "%s/labels.txt", savedir);
    printf("Saving cluster labels to file %s\n", outfile);
    FILE* fout = fopen(outfile, "w");
    for(int i = 0; i < data->ndata; i++){
        fprintf(fout, "%3d %3d\n",data->labels[i], (int)data->assigns[i]);
    }
    save_pgm_files(clust, savedir);
    

    
    for (int i = 0; i < data->nlabels; i++) {
        free(confusion[i]);
    }
    free(confusion);


    for (int i = 0; i < data->ndata; i++) {
        free(data->features[i]);
    }
    free(data->features);

    free(data->assigns);
    free(data->labels);
    free(data);

   
    for (int i = 0; i < nclust; i++) {
        free(clust->features[i]);
    }
    free(clust->features);
    free(clust->counts);

    free(clust);
    fclose(fout);



    CHECK(cudaFree(dev_data_all_feats));
    CHECK(cudaFree(dev_clust_all_feats));
    CHECK(cudaFree(dev_data_assigns));
    CHECK(cudaFree(dev_clust_counts));
    CHECK(cudaFree(nchange));
    return 0;
}