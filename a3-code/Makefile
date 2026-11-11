# CSCI 5451 A3 Makefile
AN = a3
CLASS = 5451

COPTS = -g -Wall -Wno-unused-result -O3 -std=c99 
GCC = gcc $(COPTS)
# GCC = g++ $(COPTS)
OMPCC = $(GCC) -fopenmp
NVCCOPTS = -diag-suppress 1650 -diag-suppress 2464 --compiler-options -Wno-unused-result
NVCC = nvcc -g -O3 $(NVCCOPTS)

SHELL  = /bin/bash
CWD    = $(shell pwd | sed 's/.*\///g')

PROGRAMS = \
	kmeans_serial \
	kmeans_omp \
	kmeans_cuda \

all : $(PROGRAMS)

help :
	@echo 'Typical usage is:'
	@echo '  > make                          # build all programs'
	@echo '  > make clean                    # remove all compiled items'
	@echo '  > make clean-tests              # remove all temporary testing files'
	@echo '  > make zip                      # create a zip file for submission'
	@echo '  > make prob1                    # built targets associated with problem 1'
	@echo '  > make test                     # run all tests'
	@echo '  > make test-prob2               # run test for problem 2'
	@echo '  > make test-prob2 testnum=5     # run problem 2 test #5 only'


############################################################
# 'make zip' to create p1-code.zip for submission
SHELL  = /bin/bash
CWD    = $(shell pwd | sed 's/.*\///g')
zip : clean clean-tests
	rm -f $(AN)-code.zip
	cd .. && zip "$(CWD)/$(AN)-code.zip" -r "$(CWD)"
	@echo Zip created in $(AN)-code.zip
	@if (( $$(stat -c '%s' $(AN)-code.zip) > 10*(2**20) )); then echo "WARNING: $(AN)-code.zip seems REALLY big, check there are no abnormally large test files"; du -h $(AN)-code.zip; fi
	@if (( $$(unzip -t $(AN)-code.zip | wc -l) > 256 )); then echo "WARNING: $(AN)-code.zip has 256 or more files in it which may cause submission problems"; fi


clean:
	rm -f $(PROGRAMS) *.o

################################################################################
# testing targets
test : test-prob1 test-prob2 
#test-prob3

test-setup:
	@chmod u+x testy

clean-tests :
	rm -rf test-results

################################################################################
# kmeans_serial
kmeans_serial : kmeans_serial.c kmeans_util.c
	$(GCC) -o $@ $^ -lm

test-serial: test-setup kmeans_serial
	./testy test_kmeans_serial.org $(testnum)

################################################################################
# Problem 1 : kmeans_omp
prob1: kmeans_omp

kmeans_omp : kmeans_omp.c kmeans_util.c
	$(OMPCC) -o $@ $^ -lm

test-prob1: test-setup prob1 test_kmeans_omp.org
	./testy test_kmeans_omp.org $(testnum)


################################################################################
# Problem 2 : kmeans_cuda
prob2: kmeans_cuda

kmeans_util.cu :
	$(error "No kmeans_util.cu found; try symlinking to create it as in 'ln -s kmeans_util.c kmeans_util.cu'")

kmeans_cuda : kmeans_cuda.cu kmeans_util.cu
	$(NVCC) -o $@ $^ -lm

test-prob2: test-setup prob2 test_kmeans_cuda.org
	./testy test_kmeans_cuda.org $(testnum)

