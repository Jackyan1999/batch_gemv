

#pragma once

#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define DMMA884(RD0, RD1, RA0,  RB0, RC0, RC1)     \
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64  {%0, %1}, {%2}, {%3}, {%4, %5}; \n" \
                    : "=d"(RD0), "=d"(RD1)  \
                    : "d"(RA0),  "d"(RB0),  "d"(RC0), "d"(RC1)) 

#define DMMA16816(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RA4, RA5, RA6, RA7, RB0, RB1, RB2, RB3 , RC0, RC1, RC2, RC3)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64.rn {%0, %1, %2, %3}, {%4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13,%14, %15}, {%16, %17, %18, %19};\n" \
                 : "=d"(RD0), "=d"(RD1), "=d"(RD2),"=d"(RD3)                                                                             \
                 : "d"(RA0), "d"(RA1), "d"(RA2), "d"(RA3), "d"(RA4), "d"(RA5), "d"(RA6), "d"(RA7), "d"(RB0), "d"(RB1), "d"(RB2), "d"(RB3), "d"(RC0), "d"(RC1),"d"(RC2), "d"(RC3)) 

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
