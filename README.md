# MarxAtaxx_cpp

Ataxx bot working with UAI protocol

Extract weights.zip to some folder 

Change this line(32) to reflect where you extracted the weights.zip file: const string WeightsPath="path to weights here" before compiling.

Compile with g++ -O3 -g -march=native -DNDEBUG marxataxx.cpp -o  marxataxx (requires AVX2 to work) 
