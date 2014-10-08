
/*
 * This extension is needed to store shorts in opencl
 */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

typedef struct _morientation {
    float m_orient[4];
    float m_trans[3];
    short gid;          //__attribute__ ((packed));
    short m_inverted;   //__attribute__ ((packed));
} Orientation;


typedef struct _scores {
    float fsim_raw;
    float fsim_raw_pen;
    float excl_pen;
    float vsim_raw;
} Scores;

typedef struct _overlaps {
    int count;
    short overlap[30];  //__attribute__ ((packed));
} Overlap;

typedef struct _atom {
    float x;
    float y;
    float z;
    float chge;
} Atom;

typedef struct _minfo {
    unsigned short natoms;
    unsigned short nxeds;
    unsigned short nfields;
    unsigned short norients;
} ConfInfo;

float3 rotateVec(float4 vec4, float3 vec) {
    float3 t;
    t.x = (1 - 2*vec4.y*vec4.y - 2*vec4.z*vec4.z) * vec.x + (2*vec4.x*vec4.y + 2*vec4.w*vec4.z)     * vec.y + (2*vec4.x*vec4.z - 2*vec4.w*vec4.y)     * vec.z;
    t.y = (2*vec4.x*vec4.y - 2*vec4.w*vec4.z)     * vec.x + (1 - 2*vec4.x*vec4.x - 2*vec4.z*vec4.z) * vec.y + (2*vec4.w*vec4.x + 2*vec4.y*vec4.z)     * vec.z;
    t.z = (2*vec4.x*vec4.z + 2*vec4.w*vec4.y)     * vec.x + (2*vec4.y*vec4.z - 2*vec4.w*vec4.x)     * vec.y + (1 - 2*vec4.x*vec4.x - 2*vec4.y*vec4.y) * vec.z;
    return t;
}

inline Atom translateAtom(Orientation orient, Atom a) {
    Atom b;
    float4 f = (float4) (orient.m_orient[1],orient.m_orient[2],orient.m_orient[3],orient.m_orient[0]);
    float3 trans = (float3) (orient.m_trans[0],orient.m_trans[1],orient.m_trans[2]);
    float3 vec = (float3) (a.x,a.y,a.z);
    if(orient.m_inverted) {
        vec.x = -vec.x;
    }
    vec = rotateVec(f,vec);
    vec+=trans;
    b.x = vec.x;
    b.y = vec.y;
    b.z = vec.z;
    b.chge = a.chge;
    return b;
}

__constant float rad[55]= { /*Du*/ 0.7f,
        /*H */1.0f,  /*He*/0.9f,
        /*Li*/1.2f,  /*Be*/0.9f, /*B */1.6f, /*C */1.6f, /*N*/1.4f,  /*O */1.35f, /*F */1.45f, /*Ne*/1.0f,
        /*Na*/1.5f,  /*Mg*/1.4f, /*Al*/1.2f, /*Si*/2.1f, /*P*/1.8f,  /*S */1.7f,  /*Cl*/1.75f, /*Ar*/1.9f,
        /*K */1.9f,  /*Ca*/1.9f,
        /*Sc*/1.9f,  /*Ti*/1.9f, /*V */1.9f, /*Cr*/1.9f, /*Mn*/1.9f,
        /*Fe*/1.9f,  /*Co*/1.9f, /*Ni*/1.9f, /*Cu*/1.9f, /*Zn*/1.9f,
        /*Ga*/1.9f,  /*Ge*/1.9f, /*As*/1.9f, /*Se*/1.9f, /*Br*/1.85f, /*Kr*/1.9f,
        /*Rb*/1.9f,  /*Sr*/1.9f,
        /*Y */1.9f,  /*Zr*/1.9f, /*Nb*/1.9f, /*Mo*/1.9f, /*Tc*/1.9f,
        /*Ru*/1.9f,  /*Rh*/1.9f, /*Pd*/1.9f, /*Ag*/1.9f, /*Cd*/1.9f,
        /*In*/1.9f,  /*Sn*/1.9f,
        /*Sb*/1.9f,  /*Te*/1.9f,
        /*I */1.98f, /*Xe*/2.0f
};

inline float calc_dist(Atom a0, Atom a1) {
    return (a0.x-a1.x)*(a0.x-a1.x) + (a0.y-a1.y)*(a0.y-a1.y) + (a0.z-a1.z)*(a0.z-a1.z);
}

inline float calc_rad(short nat0, short nat1, float epsilon) {
    const float f = (rad[nat0]+rad[nat1]+epsilon);
    return f*f;
}

void build_overlap_lists(__global const Atom * atoms0,__global const Atom * atoms1, __global const short * anat0,__global const short * anat1, int natoms0, int natoms1, Overlap * ovls) {
    const float epsilon = 0.0f;
    const short MAXO = 30;
    const int n = natoms0+natoms1;
    for(int i = 0; i < n; i++) {
        Atom a = i < natoms0 ? atoms0[i] : atoms1[i-natoms0];
        const short natA = i < natoms0 ? anat0[i] : anat1[i-natoms0];
        if(natA == 1) { ovls[i].count = 0; continue; } //dont count hydrogens
        ovls[i].count = 0;
        for(int j = i+1; j < n; j++) {
            Atom b =  j < natoms0 ? atoms0[j] : atoms1[j-natoms0];
            const short natB = j < natoms0 ? anat0[j] : anat1[j-natoms0];
            if(natB == 1) { continue; } //dont count hydrogens
            float dist = calc_dist(a,b);
            float radius = calc_rad(natA,natB,epsilon);
            if(dist < radius) {
                if(ovls[i].count >= MAXO) {
                    //printf("Warning: more than 30 atom overlaps found for VAtom [%d,%d]",i,j);
                    break;
                }
                ovls[i].overlap[ovls[i].count] = j;
                ovls[i].count++;
            }
        }
    }
}

void build_overlap_lists_tt(const Atom * atoms0, const Atom * atoms1, __global const short * anat0,__global const short * anat1, int natoms0, int natoms1, Overlap * ovls) {
    const float epsilon = 0.0f;
    const short MAXO = 30;
    const int n = natoms0+natoms1;
    for(int i = 0; i < n; i++) {
        Atom a = i < natoms0 ? atoms0[i] : atoms1[i-natoms0];
        const short natA = i < natoms0 ? anat0[i] : anat1[i-natoms0];
        if(natA == 1) { ovls[i].count = 0; continue; } //dont count hydrogens
        ovls[i].count = 0;
        for(int j = i+1; j < n; j++) {
            Atom b =  j < natoms0 ? atoms0[j] : atoms1[j-natoms0];
            const short natB = j < natoms0 ? anat0[j] : anat1[j-natoms0];
            if(natB == 1) { continue; } //dont count hydrogens
            float dist = calc_dist(a,b);
            float radius = calc_rad(natA,natB,epsilon);
            if(dist < radius) {
                if(ovls[i].count >= MAXO) {
                    //printf("Warning: more than 30 atom overlaps found for VAtom [%d,%d]",i,j);
                    break;
                }
                ovls[i].overlap[ovls[i].count] = j;
                ovls[i].count++;
            }
        }
    }
}


void build_overlap_lists_t(__global const Atom * atoms0, __global Atom * atoms1, __global const short * anat0,__global const short * anat1, int natoms0, int natoms1, Overlap * ovls, Orientation ortn) {
    const float epsilon = 0.0f;
    const short MAXO = 30;
    const int n = natoms0+natoms1;
    for(int i = 0; i < n; i++) {
        Atom a = i < natoms0 ? atoms0[i] : translateAtom(ortn,atoms1[i-natoms0]);
        const short natA = i < natoms0 ? anat0[i] : anat1[i-natoms0];
        if(natA == 1) { ovls[i].count = 0; continue; } //dont count hydrogens
        ovls[i].count = 0;
        for(int j = i+1; j < n; j++) {
            Atom b =  j < natoms0 ? atoms0[j] :  translateAtom(ortn,atoms1[j-natoms0]);
            const short natB = j < natoms0 ? anat0[j] : anat1[j-natoms0];
            if(natB == 1) { continue; } //dont count hydrogens
            float dist = calc_dist(a,b);
            float radius = calc_rad(natA,natB,epsilon);
            if(dist < radius) {
                if(ovls[i].count >= MAXO) {
                    //printf("Warning: more than 30 atom overlaps found for VAtom [%d,%d]",i,j);
                    break;
                }
                ovls[i].overlap[ovls[i].count] = j;
                ovls[i].count++;
            }
        }
    }
}




__kernel void mol_overlap_AB(__global const Atom * atoms0,__global const Atom * atoms1, __global const short * anat0,
 __global const short * anat1, __global const ConfInfo * cinfo, __global Orientation * orientations,
 __global float * vsimRaw, const unsigned int maxgid) {

    // get global id for this kernel
    const int gid = get_global_id(0);
    if(gid >= maxgid) {
        return;
    }

    // this kernel will work on orientation at 'gid'
    const Orientation ortn = orientations[gid];

    // get the conformation index
    const int index = ortn.gid;

    // get the number of atoms in conformation at index
    const int natoms1 = cinfo[index].natoms;

    // get a global memory pointer to the atoms
    // and atomic number for this conformation
    __global const short * ianat1 = anat1+(index*NATOMS1);
    __global const Atom * iatoms1 = atoms1 + (index*NATOMS1);

    const float p = 2.7f;
    const float pi =3.141592654f;
    const float epsilon = 0.0f;

    Overlap ovls[NATOMS0+NATOMS1];
    Atom local_atoms[8];

    float a[NATOMS0+NATOMS1];
    float volume = 0;
    float volumes[8];
    float delta[8];
    float lnKsum[8];
    short counts[8];
    short ic[8];
    short local_nat[8];

    const int n = NATOMS0+natoms1;
    const float lambda = native_divide(4*pi,p*3);
    const float avalue = native_divide(pi,native_powr(lambda,native_divide(2.0f,3.0f)));
    for(int i = 0; i < n; i++) {
        const short nat = i < NATOMS0 ? anat0[i] : ianat1[i-NATOMS0];
        if(nat == 1) { continue; }
        a[i]= native_divide(avalue,rad[nat]*rad[nat]);
    }

    build_overlap_lists_t(atoms0,iatoms1,anat0,ianat1,NATOMS0,natoms1,ovls,ortn);

    volumes[0]=-1;
    counts[0]=0;
    for(int i = 1; i < 8; i++) {
        volumes[i]=0;
        counts[i]=0;
    }
    volume = 0;
    for(int i = 0; i < NATOMS0; i++) {
        if(anat0[i] == 1) { continue; }
        local_atoms[0] = atoms0[i];
        local_nat[0] = anat0[i];
        if(rad[local_nat[0]] > 0) {
            ic[0] = i;
            delta[0] = a[i];
            lnKsum[0]=0;
            float dv = p * native_powr(native_divide(pi,delta[0]),1.5f);
            volumes[0]+=dv;
            counts[0]+=1;
            for(int j = 0; j < ovls[i].count; j++) {
                ic[1] = ovls[i].overlap[j];
                local_atoms[1] = ic[1] < NATOMS0 ? atoms0[ic[1]] : translateAtom(ortn,iatoms1[ic[1]-NATOMS0]);
                local_nat[1] = ic[1] < NATOMS0 ? anat0[ic[1]] :  ianat1[ic[1]-NATOMS0];
                Atom a0 = local_atoms[0];
                Atom a1 = local_atoms[1];
                float dist = calc_dist(a0,a1);
                delta[1]=delta[0]+a[ic[1]];
                lnKsum[1]=a[ic[0]]*a[ic[1]]*dist;
                if(ic[1] >= NATOMS0) {
                    float dv = p*p * native_exp(native_divide(-lnKsum[1],delta[1])) * native_powr(native_divide(pi,delta[1]),1.5f);
                    volume+=dv;
                    volumes[1]+=dv;
                    counts[1]+=1;
                }
#ifdef GAUSSIAN_FOUR_WAY_OVERLAPS
                for(int k = j+1; k < ovls[i].count; k++) {
                    ic[2] = ovls[i].overlap[k];
                    local_atoms[2] = ic[2] < NATOMS0 ? atoms0[ic[2]] : translateAtom(ortn,iatoms1[ic[2]-NATOMS0]);
                    local_nat[2] = ic[2] < NATOMS0 ? anat0[ic[2]] :  ianat1[ic[2]-NATOMS0];
                    Atom a0 = local_atoms[1];
                    Atom a1 = local_atoms[2];
                    if(calc_dist(a0,a1) < calc_rad(local_nat[1],local_nat[2],epsilon)) {
                        delta[2] = delta[1]+a[ic[2]];
                        lnKsum[2] = lnKsum[1];
                        for(int c = 0; c < 2; c++) {
                            Atom at = local_atoms[c];
                            float dist = calc_dist(at,a1);
                            lnKsum[2]+=a[ic[c]]*a[ic[2]]*dist;
                        }
                        if(ic[2] >= NATOMS0) {
                            float dv = native_powr(p,3) * native_exp(native_divide(-lnKsum[2],delta[2])) * native_powr(native_divide(pi,delta[2]),1.5f);
                            volume-=dv;
                            volumes[2]-=dv;
                            counts[2]+=1;
                        }
                        for(int l = k+1; l < ovls[i].count; l++) {
                            ic[3] = ovls[i].overlap[l];
                            local_atoms[3] = ic[3] < NATOMS0 ? atoms0[ic[3]] : translateAtom(ortn,iatoms1[ic[3]-NATOMS0]);
                            local_nat[3] = ic[3] < NATOMS0 ? anat0[ic[3]] :  ianat1[ic[3]-NATOMS0];
                            Atom a0 = local_atoms[1];
                            Atom a1 = local_atoms[2];
                            Atom a2 = local_atoms[3];
                            if(calc_dist(a0,a2) < calc_rad(local_nat[1],local_nat[3],epsilon) && (calc_dist(a1,a2) < calc_rad(local_nat[2],local_nat[3],epsilon))) {
                                delta[3] = delta[2]+a[ic[3]];
                                lnKsum[3] = lnKsum[2];
                                for(int c = 0; c < 3; c++) {
                                    Atom at = local_atoms[c];
                                    float dist = calc_dist(at,a2);
                                    lnKsum[3]+=a[ic[c]]*a[ic[3]]*dist;
                                }
                                if(ic[3] >= NATOMS0) {
                                    dv = native_powr(p,4) * native_exp(native_divide(-lnKsum[3],delta[3])) * native_powr(native_divide(pi,delta[3]),1.5f);
                                    volume+=dv;
                                    volumes[3]+=dv;
                                    counts[3]+=1;
                                }
#ifdef GAUSSIAN_SIX_WAY_OVERLAPS
                                for(int m = l+1; m < ovls[i].count; m++) {
                                    ic[4] = ovls[i].overlap[m];
                                    local_atoms[4] = ic[4] < NATOMS0 ? atoms0[ic[4]] : translateAtom(ortn,iatoms1[ic[4]-NATOMS0]);
                                    local_nat[4] = ic[4] < NATOMS0 ? anat0[ic[4]] :  ianat1[ic[4]-NATOMS0];
                                    Atom b = local_atoms[4];
                                    float dists[4];
                                    float rads[4];
                                    for(int c = 0; c < 4; c++) {
                                        Atom a = local_atoms[c];
                                        dists[c] = calc_dist(a,b);
                                        rads[c] = calc_rad(local_nat[c],local_nat[4],epsilon);
                                    }
                                    if((dists[1] < rads[1]) && (dists[2] < rads[2]) && (dists[3] < rads[3])) {
                                        delta[4] = delta[3]+a[ic[4]];
                                        lnKsum[4] = lnKsum[3];
                                        for(int c = 0; c < 4; c++) {
                                            lnKsum[4]+=a[ic[c]]*a[ic[4]]*dists[c];
                                        }
                                        if(ic[4] >= NATOMS0) {
                                            dv = native_powr(p,5) * native_exp(native_divide(-lnKsum[4],delta[4])) * native_powr(native_divide(pi,delta[4]),1.5f);
                                            volume-=dv;
                                            volumes[4]-=dv;
                                            counts[4]+=1;
                                        }
                                        for(int nn = m+1; nn < ovls[i].count; nn++) {
                                            ic[5] = ovls[i].overlap[nn];
                                            local_atoms[5] = ic[5] < NATOMS0 ? atoms0[ic[5]] : translateAtom(ortn,iatoms1[ic[5]-NATOMS0]);
                                            local_nat[5] = ic[5] < NATOMS0 ? anat0[ic[5]] :  ianat1[ic[5]-NATOMS0];
                                            Atom b = local_atoms[5];
                                            float dists[5];
                                            float rads[5];
                                            for(int c = 0; c < 5; c++) {
                                                Atom a = local_atoms[c];
                                                dists[c] = calc_dist(a,b);
                                                rads[c] = calc_rad(local_nat[c],local_nat[5],epsilon);
                                            }

                                            if((dists[1] < rads[1]) && (dists[2] < rads[2]) && (dists[3] < rads[3]) && (dists[4] < rads[4])) {
                                                delta[5] = delta[4]+a[ic[5]];
                                                lnKsum[5] = lnKsum[4];
                                                for(int c = 0; c < 5; c++) {
                                                    lnKsum[5]+=a[ic[c]]*a[ic[5]]*dists[c];
                                                }
                                                if(ic[5] >= NATOMS0) {
                                                    dv = native_powr(p,6) * native_exp(native_divide(-lnKsum[5],delta[5])) * native_powr(native_divide(pi,delta[5]),1.5f);
                                                    volume+=dv;
                                                    volumes[5]+=dv;
                                                    counts[5]+=1;
                                                }
#ifdef GAUSSIAN_EIGHT_WAY_OVERLAPS
                                                for(int o = nn+1; o < ovls[i].count; o++) {
                                                    ic[6] = ovls[i].overlap[o];
                                                    local_atoms[6] = ic[6] < NATOMS0 ? atoms0[ic[6]] : translateAtom(ortn,iatoms1[ic[6]-NATOMS0]);
                                                    local_nat[6] = ic[36] < NATOMS0 ? anat0[ic[6]] :  ianat1[ic[6]-NATOMS0];
                                                    Atom b = local_atoms[6];
                                                    float dists[6];
                                                    float rads[6];
                                                    for(int c = 0; c < 6; c++) {
                                                        Atom a = local_atoms[c];
                                                        dists[c] = calc_dist(a,b);
                                                        rads[c] = calc_rad(local_nat[c],local_nat[6],epsilon);
                                                    }

                                                    if((dists[1] < rads[1]) && (dists[2] < rads[2]) && (dists[3] < rads[3]) && (dists[4] < rads[4]) && (dists[5] < rads[5])) {
                                                        delta[6] = delta[5]+a[ic[6]];
                                                        lnKsum[6] = lnKsum[5];
                                                        for(int c = 0; c < 6; c++) {
                                                            lnKsum[6]+=a[ic[c]]*a[ic[6]]*dists[c];
                                                        }
                                                        if(ic[6] >= NATOMS0) {
                                                            dv = native_powr(p,7) * native_exp(native_divide(-lnKsum[6],delta[6])) * native_powr(native_divide(pi,delta[6]),1.5f);
                                                            volume-=dv;
                                                            volumes[6]-=dv;
                                                            counts[6]+=1;
                                                        }
                                                        for(int q = o+1; q < ovls[i].count; q++) {
                                                            ic[7] = ovls[i].overlap[q];
                                                            local_atoms[7] = ic[7] < NATOMS0 ? atoms0[ic[7]] : translateAtom(ortn,iatoms1[ic[7]-NATOMS0]);
                                                            local_nat[7] = ic[7] < NATOMS0 ? anat0[ic[7]] :  ianat1[ic[7]-NATOMS0];
                                                            Atom b = local_atoms[7];
                                                            float dists[7];
                                                            float rads[7];
                                                            for(int c = 0; c < 7; c++) {
                                                                Atom a = local_atoms[c];
                                                                dists[c] = calc_dist(a,b);
                                                                rads[c] = calc_rad(local_nat[c],local_nat[7],epsilon);
                                                            }

                                                            if((dists[1] < rads[1]) && (dists[2] < rads[2]) && (dists[3] < rads[3]) && (dists[4] < rads[4]) && (dists[5] < rads[5])  && (dists[6] < rads[6])) {
                                                                delta[7] = delta[6]+a[ic[7]];
                                                                lnKsum[7] = lnKsum[6];
                                                                for(int c = 0; c < 7; c++) {
                                                                    lnKsum[7]+=a[ic[c]]*a[ic[7]]*dists[c];
                                                                }
                                                                if(ic[7] >= NATOMS0) {
                                                                    dv = native_powr(p,8) * native_exp(native_divide(-lnKsum[7],delta[7])) * native_powr(native_divide(pi,delta[7]),1.5f);
                                                                    volume+=dv;
                                                                    volumes[7]+=dv;
                                                                    counts[7]+=1;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
#endif
                                            }
                                        }
                                    }
                                }
#endif
                            }
                        }
                    }
                }
#endif
            }
        }
    }
    vsimRaw[gid] = volume;
    //pritnf("%d %f\n", vsim[gid]);
}

