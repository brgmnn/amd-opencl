#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <CL/cl.h>

// These are typical values used in the simulation.
size_t size_atom    = 16,
       size_conf    = 8,
       size_orien   = 32;

unsigned int natoms_f   = 64,
             natoms_m   = 79,
             nconfs     = 25033,
             nmols      = 255,
             nkeep      = 200,
             norients   = nmols * nkeep;

// This function just fills a pointer array with junk. For the purposes of this application, it
// should be fine to do this as we don't care what the kernel computes.
void fill(void* vptr, size_t len) {
    unsigned char* ptr = (unsigned char*)vptr;
    for (int i=0; i<len; i++)
        ptr[i] = i%255;
}

int main() {
    size_t worksize = 64;
    char build_c[10000];
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint platforms, devices;

    char opt[] = "-cl-mad-enable -cl-fast-relaxed-math \
            -D NATOMS0=64 -D NXEDS0=50 -D NFIELDS0=48 \
            -D MAXCLIQUE=50 -D NKEEP=200 -D NCONS=0 \
            -D NATOMS1=79 -D NXEDS1=105 -D NFIELDS1=74 \
            -D NCONFS=25033 -D MAXFLD=600 -D MAXSCR=42 -D NFEND0=30 \
            -D GAUSSIAN_FOUR_WAY_OVERLAPS \
            -D GAUSSIAN_SIX_WAY_OVERLAPS \
            -D GAUSSIAN_EIGHT_WAY_OVERLAPS \
            ";


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

    // allocate buffers and fill with data
    unsigned char *atoms_f      = (unsigned char*)malloc(natoms_f*size_atom),
                  *atoms_m      = (unsigned char*)malloc(natoms_m*nconfs*size_atom),
                  *cinfo        = (unsigned char*)malloc(nconfs*size_conf),
                  *orients_new  = (unsigned char*)malloc(nmols*nkeep*size_orien),
                  *vsim_raw     = (unsigned char*)malloc(norients*sizeof(float));
    unsigned short *anat_f  = (unsigned short*)malloc(natoms_f*sizeof(short)),
                   *anat_m  = (unsigned short*)malloc(natoms_m*nconfs*sizeof(short));

    fill(atoms_f, natoms_f*size_atom);
    fill(atoms_m, natoms_m*nconfs*size_atom);
    fill(anat_f, natoms_f*sizeof(short));
    fill(anat_m, natoms_m*nconfs*sizeof(short));
    fill(cinfo, nconfs*size_conf);
    fill(orients_new, nmols*nkeep*size_orien);
    fill(vsim_raw, norients*sizeof(float));


    // alocate memory for buffers
    mem_atoms_f = clCreateBuffer(context, CL_MEM_READ_ONLY, natoms_f*size_atom, NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_atoms_m = clCreateBuffer(context, CL_MEM_READ_ONLY, natoms_m*nconfs*size_atom, NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_anat_f = clCreateBuffer(context, CL_MEM_READ_ONLY, natoms_f*sizeof(short), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_anat_m = clCreateBuffer(context, CL_MEM_READ_ONLY, natoms_m*nconfs*sizeof(short), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_cinfo = clCreateBuffer(context, CL_MEM_READ_WRITE, nconfs*size_conf, NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_orients_new = clCreateBuffer(context, CL_MEM_READ_WRITE, nmols*nkeep*size_orien, NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    mem_vsim_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, norients*sizeof(float), NULL, &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Create a kernel object with the compiled program
    //cl_kernel k_example=clCreateKernel(prog, "example", &error);
    cl_kernel k_example=clCreateKernel(prog, "mol_overlap_AB", &error);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Set the kernel parameters
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
    error = clSetKernelArg(k_example, 7, sizeof(unsigned int), &norients);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    unsigned int norients = 255*200;
    //vsim_AB->setArg<unsigned int>(7,norients);

    // Send input data to OpenCL (async, don't alter the buffer!)
    error=clEnqueueWriteBuffer(cq, mem_atoms_f, CL_FALSE, 0, natoms_f*size_atom, atoms_f, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_atoms_m, CL_FALSE, 0, natoms_m*nconfs*size_atom, atoms_m, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_anat_f,  CL_FALSE, 0, natoms_f*sizeof(short), anat_f, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_anat_m,  CL_FALSE, 0, natoms_m*nconfs*sizeof(short), anat_m, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_cinfo,   CL_FALSE, 0, nconfs*size_conf, cinfo, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_orients_new, CL_FALSE, 0, nmols*nkeep*size_orien, orients_new, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);
    error=clEnqueueWriteBuffer(cq, mem_vsim_raw, CL_FALSE, 0, norients*sizeof(float), vsim_raw, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // execute kernel
    error=clEnqueueNDRangeKernel(cq, k_example, 1, NULL, &worksize, &worksize, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    // Await completion of all the above
    error=clFinish(cq);
    if (error != CL_SUCCESS) printf("\n Error number %d", error);

    return 0;
}
