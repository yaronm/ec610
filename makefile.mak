# Makefile for queue simulator

# *****************************************************
# Variables to control Makefile operation

CXX = g++
CXXFLAGS = -std=c++11

# ****************************************************
# Targets needed to bring the executable up to date

main: main.o
	$(CXX) $(CXXFLAGS) -o main main.o

# The main.o target can be written more simply

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp