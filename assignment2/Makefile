CXX = icc
CXX2 = gcc
CXXFLAGS2 = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3 
#CXXFLAGS3 =
TARGET1= dot_openMP

all : $(TARGET1)
    
$(TARGET1) : dot_OpenMP.c kernel_OpenMP.c dot.h
	$(CXX) $(CXXFLAGS2) -o $(TARGET1) dot_OpenMP.c kernel_OpenMP.c -lm -fopenmp
clean : 
	rm -f $(TARGET1)
