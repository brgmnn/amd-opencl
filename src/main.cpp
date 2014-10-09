#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <CL/cl.h>

#include "types.hpp"

// values used on my nvidia machine in the working codebase
size_t globalsize = 42752;
size_t localsize  = 128;

// Read in from a file a generic primitive type. In our case this will either be floats or shorts.
template <typename T>
T* load(const char *path, const size_t size) {
    T *arr = (T*)malloc(size*sizeof(T));
    std::ifstream fp;
    fp.open(path);

    for (int i=0; i<size; i++) {
        fp >> arr[i];
    }

    fp.close();
    return arr;
}

// Read in atoms from a file.
template <>
Atom* load<Atom>(const char *path, const size_t size) {
    Atom *arr = (Atom*)malloc(size*sizeof(Atom));
    std::ifstream fp;
    fp.open(path);

    for (int i=0; i<size; i++) {
        fp >> arr[i].x;
        fp >> arr[i].y;
        fp >> arr[i].z;
        fp >> arr[i].chge;
    }

    fp.close();
    return arr;
}

// Read in ConfInfos from a file.
template <>
ConfInfo* load<ConfInfo>(const char *path, const size_t size) {
    ConfInfo *arr = (ConfInfo*)malloc(size*sizeof(ConfInfo));
    std::ifstream fp;
    fp.open(path);

    for (int i=0; i<size; i++) {
        fp >> arr[i].natoms;
        fp >> arr[i].nxeds;
        fp >> arr[i].nfields;
        fp >> arr[i].norients;
    }

    fp.close();
    return arr;
}

// Read in Orientations from a file.
template <>
Orientation* load<Orientation>(const char *path, const size_t size) {
    Orientation *arr = (Orientation*)malloc(size*sizeof(Orientation));
    std::ifstream fp;
    fp.open(path);

    for (int i=0; i<size; i++) {
        fp >> arr[i].m_orient[0];
        fp >> arr[i].m_orient[1];
        fp >> arr[i].m_orient[2];
        fp >> arr[i].m_orient[3];
        fp >> arr[i].m_trans[0];
        fp >> arr[i].m_trans[1];
        fp >> arr[i].m_trans[2];
        fp >> arr[i].gid;
        fp >> arr[i].m_inverted;
    }

    fp.close();
    return arr;
}


int main() {
    unsigned int natoms_f, natoms_m, nconfs,
                 nmols, nkeep, norients;

    Atom  *atoms_f, *atoms_m;
    short *anat_f, *anat_m;
    ConfInfo *cinfo;
    Orientation *orients_new, *orients_new_ref;
    float *vsim_raw, *vsim_raw_ref;

    char build_c[10000];
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint platforms, devices;

    // Read in the size values.
    std::ifstream fp;
    fp.open("./data/sizes.csv");
    fp >> natoms_f;
    fp >> natoms_m;
    fp >> nconfs;
    fp >> nmols;
    fp >> nkeep;
    norients = nmols*nkeep;
    fp.close();

    // Load the different buffers from CSV files.
    atoms_f = load<Atom>("./data/atoms_f.csv", natoms_f);
    atoms_m = load<Atom>("./data/atoms_m.csv", natoms_m*nconfs);
    anat_f = load<short>("./data/anat_f.csv", natoms_f);
    anat_m = load<short>("./data/anat_f.csv", natoms_m*nconfs);
    cinfo = load<ConfInfo>("./data/cinfo.csv", nconfs);
    orients_new = load<Orientation>("./data/orients_new.csv", norients);
    vsim_raw = load<float>("./data/vsim_raw.csv", norients);

    orients_new_ref = load<Orientation>("./data/orients_new_ref.csv", norients);
    vsim_raw_ref = load<float>("./data/vsim_raw_ref.csv", norients);

    // Fetch the Platforms, we only want one.
    error=clGetPlatformIDs(1, &platform, &platforms);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Fetch the Devices for this platform
    error=clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &devices);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Create a memory context for the device we want to use
    cl_context_properties properties[]={CL_CONTEXT_PLATFORM, (cl_context_properties)platform,0};
    cl_context context=clCreateContext(properties, 1, &device, NULL, NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Create a command queue to communicate with the device
    cl_command_queue cq = clCreateCommandQueue(context, device, 0, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // set all params for kernels
    char opt[200];
    sprintf(opt, "-cl-mad-enable -cl-fast-relaxed-math -D NATOMS0=%d -D NATOMS1=%d -D NCONFS=%d \
            -D NKEEP=%d -D GAUSSIAN_FOUR_WAY_OVERLAPS -D GAUSSIAN_SIX_WAY_OVERLAPS \
            -D GAUSSIAN_EIGHT_WAY_OVERLAPS", natoms_f, natoms_m, nconfs, nkeep);

    // read in the kernel source
    std::ifstream t("kernel/volume-ab.cl");
    std::stringstream sst;
    sst << t.rdbuf();
    std::string src = sst.str();
    const char* csrc = src.c_str();
    const size_t srclen = src.length();


    // Submit the source code of the kernel to OpenCL, and create a program object with it
    cl_program prog=clCreateProgramWithSource(context, 1, &csrc, &srclen, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Compile the kernel code (after this we could extract the compiled version)
    error=clBuildProgram(prog, 0, NULL, opt, NULL, NULL);
    if ( error != CL_SUCCESS ) {
        printf( "Error on buildProgram " );
        printf("\n Error number %d", error);
        fprintf( stdout, "\nRequestingInfo\n" );
        clGetProgramBuildInfo( prog, device, CL_PROGRAM_BUILD_LOG, 4096, build_c, NULL );
        printf( "Build Log for %s_program:\n%s\n", "example", build_c );
    }

    /* Create memory buffers in the Context where the desired Device is. These will be the pointer
    parameters on the kernel. */
    cl_mem mem_atoms_f, mem_atoms_m, mem_anat_f, mem_anat_m, mem_cinfo,
           mem_orients_new, mem_vsim_raw;

    // alocate memory for buffers
    mem_atoms_f = clCreateBuffer(context, CL_MEM_READ_ONLY, natoms_f*sizeof(Atom), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_atoms_m = clCreateBuffer(context, CL_MEM_READ_ONLY, natoms_m*nconfs*sizeof(Atom), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_anat_f = clCreateBuffer(context, CL_MEM_READ_ONLY, natoms_f*sizeof(short), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_anat_m = clCreateBuffer(context, CL_MEM_READ_ONLY, natoms_m*nconfs*sizeof(short), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_cinfo = clCreateBuffer(context, CL_MEM_READ_WRITE, nconfs*sizeof(ConfInfo), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_orients_new = clCreateBuffer(context, CL_MEM_READ_WRITE, nmols*nkeep*sizeof(Orientation), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_vsim_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, norients*sizeof(float), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Create a kernel object with the compiled program
    cl_kernel k_example=clCreateKernel(prog, "mol_overlap_AB", &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Set the kernel parameters
    unsigned int unorients = norients;
    error = clSetKernelArg(k_example, 0, sizeof(mem_atoms_f), &mem_atoms_f);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error = clSetKernelArg(k_example, 1, sizeof(mem_atoms_m), &mem_atoms_m);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error = clSetKernelArg(k_example, 2, sizeof(mem_anat_f), &mem_anat_f);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error = clSetKernelArg(k_example, 3, sizeof(mem_anat_m), &mem_anat_m);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error = clSetKernelArg(k_example, 4, sizeof(mem_cinfo), &mem_cinfo);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error = clSetKernelArg(k_example, 5, sizeof(mem_orients_new), &mem_orients_new);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error = clSetKernelArg(k_example, 6, sizeof(mem_vsim_raw), &mem_vsim_raw);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error = clSetKernelArg(k_example, 7, sizeof(unsigned int), &unorients);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Send input data to OpenCL (async, don't alter the buffer!)
    error=clEnqueueWriteBuffer(cq, mem_atoms_f, CL_FALSE, 0, natoms_f*sizeof(Atom), atoms_f, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_atoms_m, CL_FALSE, 0, natoms_m*nconfs*sizeof(Atom), atoms_m, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_anat_f,  CL_FALSE, 0, natoms_f*sizeof(short), anat_f, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_anat_m,  CL_FALSE, 0, natoms_m*nconfs*sizeof(short), anat_m, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_cinfo,   CL_FALSE, 0, nconfs*sizeof(ConfInfo), cinfo, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_orients_new, CL_FALSE, 0, nmols*nkeep*sizeof(Orientation), orients_new, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_vsim_raw, CL_FALSE, 0, norients*sizeof(float), vsim_raw, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // execute kernel
    error=clEnqueueNDRangeKernel(cq, k_example, 1, NULL, &globalsize, &localsize, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clFinish(cq);

    /* Read the result back into buf2 */
    error=clEnqueueReadBuffer(cq, mem_orients_new, CL_FALSE, 0, nmols*nkeep*sizeof(Orientation), orients_new, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueReadBuffer(cq, mem_vsim_raw, CL_FALSE, 0, norients*sizeof(float), vsim_raw, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Await completion of all the above
    error=clFinish(cq);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    //for (int i=0; i<10; i++)
    //    std::cout << vsim_raw[i] << " - " << vsim_raw_ref[i] << std::endl;

    return 0;
}
