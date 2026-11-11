#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
// #include "filestats.c"

typedef struct KMData KMData;
typedef struct KMClust KMClust;

struct KMData {
    int ndata;
    int dim;
    float **features;
    int *assigns;
    int *labels;
    int nlabels;
};

struct KMClust {
    int nclust;
    int dim;
    float **features;
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

KMData *kmdata_load(char *datafile) {
    ssize_t toks;
    ssize_t lines;
    filestat(datafile,&toks,&lines);
    KMData *data = (KMData *)malloc(sizeof(KMData));

    data->ndata = (int)lines;
    int numtoks = (int)toks/(int)lines;
    data->dim = numtoks-2;
    // printf("dim: %d\n", data->dim);
    // printf("ndatata: %d\n", data->ndata);

    data->features = (float**)malloc(data->ndata * sizeof(float *));
    for (int i = 0; i < data->ndata; i++) {
        data->features[i] = (float*)malloc(data->dim * sizeof(float));
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

KMClust *kmclust_new(int nclust, int dim) {
    KMClust *clust = (KMClust *)malloc(sizeof(KMClust));
    clust->nclust = nclust;
    clust->dim = dim;
    clust->features = malloc(nclust * sizeof(float *));
    for (int i = 0; i < nclust; i++) {
        clust->features[i] = malloc(dim * sizeof(float));
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





int main(int argc, char **argv){
    if(argc<3){
        printf("error");
        return -1;
    }

    char* datafile = (char*)argv[1];
    int nclust = atoi(argv[2]);
    int MAXITER =100;
    char *savedir = ".";

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

    KMData *data = kmdata_load(datafile);
    KMClust *clust = kmclust_new(nclust, data->dim);

    printf("ndata: %d\n",data->ndata);
    printf("dim: %d\n\n",data->dim);

    int c;
    for(int i = 0; i < data->ndata; i++){
        c = i%clust->nclust;
        data->assigns[i] = c;
    }

    for(int c = 0; c < clust->nclust; c++){
        int icount = round(data->ndata/clust->nclust);
        int extra = 0;
        if(c < (data->ndata % clust->nclust)){
            extra = 1;
        }
        clust->counts[c] = icount + extra;
    }

    int curiter = 1;                       
    int nchanges = data->ndata;               
    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");
    while(nchanges > 0 && curiter <= MAXITER){
        for(int c = 0; c < clust->nclust; c++){
            for(int d = 0; d < clust->dim; d++){
                clust->features[c][d] = 0.0;
            }
        }
        
        for(int i = 0; i < data->ndata; i++){
            int c = data->assigns[i];
            for(int d = 0; d < clust->dim; d++){
                clust->features[c][d] += data->features[i][d];
            }
        }
        for(int c = 0; c < clust->nclust; c++){
            if(clust->counts[c]>0){
                for(int d = 0;d<clust->dim;d++){
                    clust->features[c][d] = clust->features[c][d]/clust->counts[c];
                }
            }
        }
        for(int c=0;c<clust->nclust;c++){
            clust->counts[c]=0;
        }

        nchanges = 0;
        for(int i = 0; i<data->ndata;i++){
            int best_clust = 0;
            float best_distsq = INFINITY;
            for(int c=0;c<clust->nclust;c++){
                float distsq =0.0;
                for(int d=0;d<clust->dim;d++){
                    float diff= data->features[i][d] - clust->features[c][d];
                    distsq += diff*diff;
                    // printf("DBG: curiter %d i %d c %d d %d diff %.6f\n",curiter,i,c,d,diff);

                }
                if(distsq < best_distsq){
                    best_clust = c;
                    best_distsq = distsq;
                }
            }
            clust->counts[best_clust] +=1;
            if(best_clust != data->assigns[i]){
                // printf("DBG: data %d changing clusters: %d to %d\n",i,data->assigns[i],best_clust);
                nchanges++;
                data->assigns[i] = best_clust;
            }
        }
        printf("%4d:  %4d | ",curiter,nchanges);
        for(int c=0;c<nclust;c++){
            printf("%4d  ",clust->counts[c]);
        }
        printf("\n");
        curiter++;
    }

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
        for (int j = 0; j < clust->nclust; j++) {
            confusion[i][j] = 0;
        }
    }
    for(int i = 0;i<data->ndata;i++){
        confusion[data->labels[i]][data->assigns[i]]++;
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
        printf("%5d",clust->counts[c]);
        tot = tot + clust->counts[c];
    }
    printf(" %4d\n\n",tot);


    char outfile[256];
    sprintf(outfile, "%s/labels.txt", savedir);
    printf("Saving cluster labels to file %s\n", outfile);
    FILE* fout = fopen(outfile, "w");
    for(int i = 0; i < data->ndata; i++){
        fprintf(fout, "%3d %3d\n",data->labels[i], data->assigns[i]);
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
    return 0;
}