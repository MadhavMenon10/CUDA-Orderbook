.PHONY: clean
CXXFLAGS_CPU = -std=c++23 -Iincludes/ -I/usr/local/cuda/include/ -Wall -Wextra -pedantic 
CXXFLAGS_GPU = -std=c++20 -Iincludes/ 
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

bin/itch_parser: build/reader.o build/decoder.o build/soa.o build/main.o build/hash_table.o build/symbol_compactor.o build/reconstructor.o build/backtester.o build/results_analysis.o
	mkdir -p bin
	nvcc $(LDFLAGS) $^ -o $@

# Compiles each src file into its object file in the build directory
build/%.o: src/%.cpp
	mkdir -p build 
	$(CXX) $(CXXFLAGS_CPU) -c $< -o $@

build/%.o: src/%.cu
	mkdir -p build 
	nvcc $(CXXFLAGS_GPU) -c $< -o $@

exec: bin/itch_parser
	./bin/itch_parser

clean:
	rm -rf bin/* build/*




