# Makefile for queue simulator

# *****************************************************
# Variables to control Makefile operation

CXX = g++
CXXFLAGS = -std=c++11 -O3

# ****************************************************
# Targets needed to bring the executable up to date

all: run
main: main.o
	$(CXX) $(CXXFLAGS) -o main main.o



main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

run: main
	./main
.PHONY: all run
