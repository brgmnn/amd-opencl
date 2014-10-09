inc = -I/opt/AMDAPP/include
lib = -lOpenCL
flags = -O3 -g -funroll-loops -finline-functions -mfpmath=sse

all:
	g++ $(flags) $(inc) src/main.cpp -o cltest $(lib)

clean:
	rm cltest
