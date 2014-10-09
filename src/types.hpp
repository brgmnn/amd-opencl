#pragma once

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

typedef struct _morientation {
    float m_orient[4];
    float m_trans[3];
    short gid;          //__attribute__ ((packed));
    short m_inverted;   //__attribute__ ((packed));
} Orientation;
