inc = -I/opt/AMDAPP/include
lib = -lOpenCL

all:
	g++ $(inc) src/main.cpp -o cltest $(lib)

clean:
	rm cltest
