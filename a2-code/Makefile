# CSCI 5451 A2 Makefile
AN = a2
CLASS = 5451

COPTS = -g -Wall -O3 -std=c99
GCC = gcc $(COPTS)
MPICC = mpicc $(COPTS)

SHELL  = /bin/bash
CWD    = $(shell pwd | sed 's/.*\///g')

PROGRAMS = \
	mpi_hello \
	heat_serial \
	heat_mpi    \
	kmeans_serial \
	kmeans_mpi \

# kmeans_omp \
# dense_pagerank_serial \
# dense_pagerank_mpi \
# dense_pagerank_omp \
# dense_pagerank_blas \


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
# Demo MPI program
mpi_hello : mpi_hello.c
	$(MPICC) -o $@ $^

################################################################################
# Problem 1 : heat
prob1: heat_serial heat_mpi

heat_serial : heat_serial.c
	$(GCC) -o $@ $^

heat_mpi : heat_mpi.c
	$(MPICC) -o $@ $^

test-prob1: test-setup heat_mpi
	./testy test_heat.org $(testnum)

################################################################################
# Problem 2 : kmeans_serial

prob2: kmeans_serial

kmeans_serial : kmeans_serial.c kmeans_util.c
	$(GCC) -o $@ $^ -lm

test-prob2: test-setup prob2 
	./testy test_kmeans_serial.org $(testnum)

################################################################################
# Problem 3 : kmeans_mpi
prob3: kmeans_mpi

kmeans_mpi : kmeans_mpi.c kmeans_util.c
	$(MPICC) -o $@ $^ -lm

test-prob3: test-setup prob3 test_kmeans_mpi.org
	./testy test_kmeans_mpi.org $(testnum)


