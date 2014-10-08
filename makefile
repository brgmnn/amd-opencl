inc = -I/opt/AMDAPP/include
lib = -lOpenCL

all:
	clang++ $(inc) src/main.cpp -o cltest $(lib)

clean:
	rm cltest
