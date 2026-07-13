.PHONY: clean
CXXFLAGS = -std=c++23 -Iincludes/ -I/usr/local/cuda/includes/ -Wall -Wextra -pedantic 
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -rf bin/* build/*

# Compiles each src file into its object file in the build directory
build/%.o: src/%.cpp
	mkdir -p build 
	$(CXX) $(CXXFLAGS) -c $< -o $@


bin/itch_parser: build/reader.o build/decoder.o build/soa.o build/main.o
	mkdir -p bin
	$(CXX) $(LDFLAGS) $^ -o $@

