# MarxAtaxx_cpp

Ataxx bot working with UAI protocol

Extract weights.zip to some folder 

Change line 67 to reflect where you extracted the weights.zip file: const string WeightsPath="path to weights here" before compiling.

By default it will run with 8 threads, you can change this by changing line 1620 const int NTHREADS = 8; to whatever suits you (only tested with 8)

Compile with g++ -O3 -g -march=native -DNDEBUG marxataxx.cpp -o  marxataxx (requires AVX2 to work) 
