#include <stdio.h>  
#include <stdlib.h>  
#include <cuda.h>  
#include<cuda_fp16.h>
// #include <ctime.h>
#include <cublas_v2.h>  
  
#define CHECK_CUDA(call) {                                 \
    const cudaError_t error = call;                        \
    if (error != cudaSuccess) {                            \
        printf("Error: %s:%d, ", __FILE__, __LINE__);      \
        printf("CUDA error: %s\n", cudaGetErrorString(error)); \
        exit(EXIT_FAILURE);                               \
    }                                                     \
}  
  
#define CHECK_CUBLAS(call) {                                      \
    cublasStatus_t status = call;                              \
    if (status != CUBLAS_STATUS_SUCCESS) {               \
        printf("Error: %s:%d, ", __FILE__, __LINE__);    \
        printf("cuBLAS error: %d\n", status);            \
        cublasDestroy(handle);                           \
        exit(EXIT_FAILURE);                              \
    }                                                    \
}  


// #define MMA_M 16
// #define MMA_N 8
// #define MMA_K 16

// __global__ void batchedgemv_kernel(size_t m,size_t n, __half** d_Aarray,size_t lda, 
//                                                 __half** d_xarray,__half** d_yarray,size_t batchCount)
// {

//     __half* A = d_Aarray[blockIdx.x];
//     __half* x = d_xarray[blockIdx.x];

//     extern __shared__ half smem[][n];

//     const size_t M_tiles = div_ceil(M, MMA_M);
//     const size_t N_tiles = div_ceil(N, MMA_N);
//     const size_t K_tiles = div_ceil(K, MMA_K);


//     const size_t warp_id = threadIdx.x / WARP_SIZE;
//     const size_t lane_id = threadIdx.x % WARP_SIZE;


    
// }


#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32

__device__ float f16_to_f32(half __x) {
    unsigned short n = *((unsigned short *)&__x);
    unsigned int x = (unsigned int)n;
    x = x & 0xffff;
    unsigned int sign = x & 0x8000;                   //符号位
    unsigned int exponent_f16 = (x & 0x7c00) >> 10;   //half指数位
    unsigned int mantissa_f16 = x & 0x03ff;           //half小数位
    unsigned int y = sign << 16;
    unsigned int exponent_f32;                        //float指数位
    unsigned int mantissa_f32;                        //float小数位
    unsigned int first_1_pos = 0;                     //（half小数位）最高位1的位置
    unsigned int mask;
    unsigned int hx;
   
    hx = x & 0x7fff;
   
    if (hx == 0) {
      return *((float *)&y);
    }
    if (hx == 0x7c00) {
      y |= 0x7f800000;
      return *((float *)&y);
    }
    if (hx > 0x7c00) {
      y = 0x7fc00000;
      return *((float *)&y);
    }
   
    exponent_f32 = 0x70 + exponent_f16;
    mantissa_f32 = mantissa_f16 << 13;
   
    for (first_1_pos = 0; first_1_pos < 10; first_1_pos++) {
      if ((mantissa_f16 >> (first_1_pos + 1)) == 0) {
        break;
      }
    }
   
    if (exponent_f16 == 0) {
      mask = (1 << 23) - 1;
      exponent_f32 = exponent_f32 - (10 - first_1_pos) + 1;
      mantissa_f32 = mantissa_f32 << (10 - first_1_pos);
      mantissa_f32 = mantissa_f32 & mask;
    }
   
    y = y | (exponent_f32 << 23) | mantissa_f32;
   
    return *((float *)&y);
  }



 float f16_to_f32_cpu(half __x) {
    unsigned short n = *((unsigned short *)&__x);
    unsigned int x = (unsigned int)n;
    x = x & 0xffff;
    unsigned int sign = x & 0x8000;                   //符号位
    unsigned int exponent_f16 = (x & 0x7c00) >> 10;   //half指数位
    unsigned int mantissa_f16 = x & 0x03ff;           //half小数位
    unsigned int y = sign << 16;
    unsigned int exponent_f32;                        //float指数位
    unsigned int mantissa_f32;                        //float小数位
    unsigned int first_1_pos = 0;                     //（half小数位）最高位1的位置
    unsigned int mask;
    unsigned int hx;
   
    hx = x & 0x7fff;
   
    if (hx == 0) {
      return *((float *)&y);
    }
    if (hx == 0x7c00) {
      y |= 0x7f800000;
      return *((float *)&y);
    }
    if (hx > 0x7c00) {
      y = 0x7fc00000;
      return *((float *)&y);
    }
   
    exponent_f32 = 0x70 + exponent_f16;
    mantissa_f32 = mantissa_f16 << 13;
   
    for (first_1_pos = 0; first_1_pos < 10; first_1_pos++) {
      if ((mantissa_f16 >> (first_1_pos + 1)) == 0) {
        break;
      }
    }
   
    if (exponent_f16 == 0) {
      mask = (1 << 23) - 1;
      exponent_f32 = exponent_f32 - (10 - first_1_pos) + 1;
      mantissa_f32 = mantissa_f32 << (10 - first_1_pos);
      mantissa_f32 = mantissa_f32 & mask;
    }
   
    y = y | (exponent_f32 << 23) | mantissa_f32;
   
    return *((float *)&y);
  }

//
__global__ void check_kernel(size_t batchCount, size_t m , size_t n , __half* d_A, __half* d_x, __half* d_y, __half* d_y_tc)
{
  for( int b = 0; b < batchCount; b++)
  {
    for( int i = 0; i < m; i++)
    {
      half tmp = 0.0f;
      for(int j = 0; j < n; j++)
      {
        tmp += d_A[b * m * n + i * n + j] * d_x[b * n + j];
      }
      d_y[b * m + i] = tmp;
      // printf("%f \n", __half2float(tmp));
    }

  }
    half err_line = 0.00001f;
    //check
    for(int i = 0 ; i < batchCount * m;  i++)
    {
        if((d_y[i] - d_y_tc[i]) > err_line || d_y_tc[i] - d_y[i] > err_line)
        {
          printf("gr: i: %d\n" , i);
        }
        printf("VAL: check: %f , my: %f\n", __half2float(d_y[i]), __half2float(d_y_tc[i]));

    }

}


// dim3 block(WARP_SIZE);
// dim3 grid(batchCount, div_ceil(M, MMA_M));
__global__ void batchedgemv_kernel(size_t m,size_t n, __half** d_Aarray,size_t lda, 
                                                __half** d_xarray,__half** d_yarray,size_t batchCount) 
{

    //printf("laaaaaaaaaaaaaaaaaaaaaaaaaaalllllllllllllllllllllllllllaaaaaaaaaaaaaaaaaaaaaa\n");
    const size_t N_tiles = div_ceil(n, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x ;         //batchcount

    // if (warp_row >= M || warp_col >= N) {
    //     return;
    // }

    __shared__ half A_smem[MMA_M][MMA_K];
    // __shared__ half B_smem[MMA_N][MMA_K];
    // __shared__ half C_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[2] = {0, 0};
    uint32_t RA[4] = {0, 0, 0, 0};
    uint32_t RB[2] = {0, 0};

    half* A = d_Aarray[blockIdx.x];
    // printf("ccccccccccccccccccccccccccccccccccccccccccccccccccc\n");
#pragma unroll
    for (size_t i = 0; i < N_tiles; ++i) {

        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * n + i * MMA_K]) + lane_id % 2);

        // if(threadIdx.x==0)
        // {
        //   printf("lane_id : %d ,  A_POS: %ld \n",threadIdx.x, ((warp_row + lane_id / 2) * n + i * MMA_K));
        // }
        // if (lane_id < MMA_N * 2) {
        //     *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
        //         *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        // }
        // printf("lllllllllllllllllllllllllliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n");
        // B 
        size_t Thread_pos_1 = lane_id * 2 ;
        size_t Thread_pos_2 = lane_id * 2 + 8 ;
        if(lane_id < 4)
        {
            RB[0] = *((uint32_t*)(d_xarray[warp_col]  + Thread_pos_1));
            RB[1] = *((uint32_t*)(d_xarray[warp_col]  + Thread_pos_2));
            // printf("threadIdx: %d , %f  \n", threadIdx.x,  f16_to_f32(*((half*)(&RB[0]))));
            // printf("threadIdx: %d , %f  \n", threadIdx.x,  f16_to_f32(*((half*)(&RB[0])+1)));
            // printf("threadIdx: %d , %f  \n", threadIdx.x,  f16_to_f32(*((half*)(&RB[1]))));
            // printf("threadIdx: %d , %f \n", threadIdx.x, f16_to_f32(*((half*)(&RB[1])+1)));
        }

        //printf("dddddddddddddddddddddddddddddddddddddddddddddddddddddd\n");
        __syncthreads();
        // for(int i  = 0 ;i < MMA_M * MMA_N; i++)
        // {
        //     float tmp = f16_to_f32(A_smem[i]);
        //     printf("%f ",tmp);
        //     if(i % MMA_N) printf("\n");
        // }

        // printf("i : %d\n" ,i);
        // if(threadIdx.x ==0)
        // {
        //   for(int i  = 0 ;i < MMA_M ; i++)
        //   {
        //       for( int j = 0 ;j < MMA_K ;j++)
        //       {
        //               float tmp = f16_to_f32(A_smem[i][j]);
        //               printf("%f ",tmp);
                   
        //       }
        //        printf("\n");
        //   }
        // }

        
        // printf("goooogooooooooooooooooooooooooooooooooooooooooooooooooooooog\n");
        // uint32_t RA[4];
        // uint32_t RB[2];

        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        // uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        // LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    // *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    // *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    // __syncthreads();

    // if (lane_id < MMA_M) {
    //     *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    // }
    //printf("fuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuk\n");
    size_t Thread_pos_1 = lane_id / 4;
    size_t Thread_pos_2 = lane_id / 4 + 8;

    if(lane_id % 4 ==0)
    {
        *(d_yarray[warp_col]  + Thread_pos_1 + blockIdx.y * 16 ) = *(half*)(&(RC[0]));
        *(d_yarray[warp_col]  + Thread_pos_2 + blockIdx.y * 16) = *(half*)(&(RC[1]));
        // *(d_yarray[warp_col]  + Thread_pos_1) = half(RC[0]);
        // *(d_yarray[warp_col]  + Thread_pos_2) = half(RC[1]);
        // printf("RC0 : %f \n" , __half2float(*(half*)(&(RC[0]))));
        // printf("RC1 : %f \n" , __half2float(*(half*)(&(RC[1]))));
        // if(lane_id==8)
        // {
        //     half* p = (half*)(&(RC[1])) ;
        //     half tmp = *p;
        //     float tmp_1 = f16_to_f32(tmp);
        //     printf("%f \n", tmp_1);
        // }

    }
    // if(lane_id==0)
    // {
    //   for(int i = 0; i< m ;i++)
    //   {
    //        printf("d_yarray[%d] : %f \n", i, __half2float(*(d_yarray[warp_col] + i)));
    //   }
      
    // }
    //printf("Here!!!!!!!!!!!!!\n");

}

// void mmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
//     dim3 block(WARP_SIZE);
//     dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

//     mmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
// }


int main() {  
    cublasHandle_t handle;  
    float h_alpha = 1.0f;  
    float h_beta = 0.0f;  
    int m = 128; // 矩阵的行数  
    int n = 128; // 矩阵的列数  
    int lda = m; // 矩阵A的领先维度  
    int batchCount = 512; // 批处理大小  
    __half* h_A = new __half[m * n * batchCount];
    __half* h_x = new __half[n * batchCount];
    __half* h_y = new __half[m * batchCount];
    __half* h_y_blas = new __half[m * batchCount];


     srand((unsigned)time(0));

     for(int i = 0;i < m * n * batchCount;i++)
     {
         h_A[i] = ((float)i) / 100000;
        //  printf("%f \n", f16_to_f32_cpu(h_A[i]));
     }
     for(int i = 0;i < n * batchCount;i++)
     {
         h_x[i] = ((float)i) / 100000;
     }
     for(int i = 0;i < m * batchCount;i++)
     {
         h_y[i] = 3;
     }



    //  h_A[i].r = double(rand())/RAND_MAX;
    //  h_A[i].i = double(rand())/RAND_MAX;
    // __half RAND_MAX = 100.0f;

    // 假设h_A和h_x已经被初始化了一些值（在实际代码中你需要做这一步）  
    // for(int i = 0;i < m * n * batchCount;i++)
    // {
    //     h_A[i] = 1;
    // }
    // for(int i = 0;i < n * batchCount;i++)
    // {
    //     h_x[i] = 2;
    // }
    // for(int i = 0;i < m * batchCount;i++)
    // {
    //     h_y[i] = 3;
    // }
    // printf("%f \n", f16_to_f32_cpu(half(double(rand())/(180428938.f)))); 

    // for(int i = 0;i < m * n * batchCount;i++)
    // {
    //     h_A[i] = half(double(rand())/(1804289380.f));
    //     // printf("%f \n", f16_to_f32_cpu(h_A[i]));
        
    // }
    // for(int i = 0;i < n * batchCount;i++)
    // {
    //     h_x[i] = half(double(rand())/(1804289380.f));
    // }
    // for(int i = 0;i < m * batchCount;i++)
    // {
    //     h_y[i] = half(double(rand())/(180428938.f));
    // }
    // for(int i = 0;i < m * batchCount;i++)
    // {
    //     h_y_blas[i] = half(double(rand())/(180428938.f));
    // }

    // for(int i = 0;i < m * n * batchCount;i++)
    // {
    //     h_A[i] = half(i);
    //     // printf("%f \n", f16_to_f32_cpu(h_A[i]));
        
    // }
    // for(int i = 0;i < n * batchCount;i++)
    // {
    //     h_x[i] = half(i);
    // }
    // for(int i = 0;i < m * batchCount;i++)
    // {
    //     h_y[i] = half(i);
    // }
    // for(int i = 0;i < m * batchCount;i++)
    // {
    //     h_y_blas[i] = 0.0f;
    //     // printf("%f \n",f16_to_f32_cpu(h_y_blas[i]));
    // }

    // 创建设备指针  
    __half *Aarray[batchCount];  
    __half *xarray[batchCount];  
    __half *yarray[batchCount];  
    __half *d_A, *d_x, *d_y ,*d_y_test;  
   // printf("0000000000000000000000000000000000000\n");
    // 初始化cuBLAS  
    CHECK_CUBLAS(cublasCreate(&handle));  
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));  
  
    // 分配设备内存  
    // CHECK_CUDA(cudaMalloc((void **)&d_A, m * n * batchCount * sizeof(__half)));  
    // CHECK_CUDA(cudaMalloc((void **)&d_x, n * batchCount * sizeof(__half)));  
    // CHECK_CUDA(cudaMalloc((void **)&d_y, m * batchCount * sizeof(__half)));  
    CHECK_CUDA(cudaMalloc(&d_A, m * n * batchCount * sizeof(__half)));  
    CHECK_CUDA(cudaMalloc(&d_x, n * batchCount * sizeof(__half)));  
    CHECK_CUDA(cudaMalloc(&d_y, m * batchCount * sizeof(__half)));  
    CHECK_CUDA(cudaMalloc(&d_y_test, m * batchCount * sizeof(__half)));  
    // 计算每个批处理元素的设备指针  
    for (int i = 0; i < batchCount; ++i) {  
        Aarray[i] = d_A + i * m * n;  
        xarray[i] = d_x + i * n;  
        yarray[i] = d_y + i * m;  
    }  
   // printf("111111111111111111111111111111111111111\n");
    __half **d_Aarray;
    __half **d_xarray;
    __half **d_yarray;

    CHECK_CUDA(cudaMalloc((void **)&d_Aarray, batchCount * sizeof(__half*)));  
    CHECK_CUDA(cudaMalloc((void **)&d_xarray, batchCount * sizeof(__half*)));  
    CHECK_CUDA(cudaMalloc((void **)&d_yarray, batchCount * sizeof(__half*)));  
    // 将数据从主机复制到设备  
    CHECK_CUDA(cudaMemcpy(d_A, h_A, m * n * batchCount * sizeof(__half), cudaMemcpyHostToDevice));  
    CHECK_CUDA(cudaMemcpy(d_x, h_x, n * batchCount * sizeof(__half), cudaMemcpyHostToDevice));  
    CHECK_CUDA(cudaMemcpy(d_Aarray, Aarray, batchCount * sizeof(__half*), cudaMemcpyHostToDevice));  
    CHECK_CUDA(cudaMemcpy(d_xarray, xarray, batchCount * sizeof(__half*), cudaMemcpyHostToDevice));  
    CHECK_CUDA(cudaMemcpy(d_yarray, yarray, batchCount * sizeof(__half*), cudaMemcpyHostToDevice));  
   // printf("22222222222222222222222222222222222222\n");
    // 调用cublasHSHgemvBatched  
    // for(int i = 0;i < 1024 ;i++)
    // {
    // CHECK_CUBLAS(cublasHSHgemvBatched(handle, CUBLAS_OP_N, m, n,  
    //                                   &h_alpha, d_Aarray, lda, d_xarray, 1,  
    //                                   &h_beta, d_yarray, 1, batchCount));  
   
   
#define time    
#ifdef time 
    float msecTotal_blas;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 block(WARP_SIZE);
    dim3 grid(batchCount, div_ceil(m, MMA_M));
    // for(int i = 0; i < 100; i++)
    // {
    //   batchedgemv_kernel<<<grid,block>>>(m,n,d_Aarray,lda,d_xarray,d_yarray,batchCount);
    // }
    cublasStatus_t status;
    //warm_up
    for(int i = 0;i < 1000; i++)
    {
      status = cublasHSHgemvBatched(handle, CUBLAS_OP_T, m, n,  
        &h_alpha, d_Aarray, lda, d_xarray, 1,  
        &h_beta, d_yarray, 1, batchCount);  
    }
    cudaDeviceSynchronize();



    

    CHECK_CUDA(cudaEventRecord(start));    
    for(int i = 0;i < 100; i++)
    {
      status = cublasHSHgemvBatched(handle, CUBLAS_OP_T, m, n,  
        &h_alpha, d_Aarray, lda, d_xarray, 1,  
        &h_beta, d_yarray, 1, batchCount);  
    }
    cudaDeviceSynchronize();
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal_blas, start, stop));
    printf("blas:%d\n",status);
    CHECK_CUDA(cudaMemcpy(h_y_blas, d_y, m * batchCount * sizeof(__half), cudaMemcpyDeviceToHost));  


#endif


//     printf("blas:%d\n",status);
//    // }
//     printf("3333333333333333333333333333333333333333\n");

//     cudaDeviceSynchronize();
    

    // size_t sharedmem_size = (m*n+8*n)*sizeof(__half);


#ifdef time 
    float msecTotal_me;
    // dim3 grid(batchCount);
    // dim3 block(128);
    CHECK_CUDA(cudaEventRecord(start));   
#endif

      for(int i = 0;i < 100 ;i++)
      {
        batchedgemv_kernel<<<grid,block>>>(m,n,d_Aarray,lda,d_xarray,d_yarray,batchCount);
      }

      cudaDeviceSynchronize();
#ifdef time 
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

   
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal_me, start, stop));

 
    printf("blas_time : %f \n",msecTotal_blas);
    printf("my_time : %f \n", msecTotal_me);
    printf("blas_time/my_time %f \n" , msecTotal_blas / msecTotal_me);
#endif
    //将结果从设备复制回主机  
  
    CHECK_CUDA(cudaMemcpy(h_y, d_y, m * batchCount * sizeof(__half), cudaMemcpyDeviceToHost));  
    // 此处可以添加代码来处理或使用h_y中的结果  
  
    // 清理  
    // CHECK_CUDA(cudaFree(d_A));  
    // CHECK_CUDA(cudaFree(d_x));  
    // CHECK_CUDA(cudaFree(d_y));  
   // printf("444444444444444444444444444444444444444444\n");

    // CHECK_CUBLAS(cublasDestroy(handle));  
    
    // CHECK_CUDA(cudaMemcpy(h_y, d_y, m * batchCount * sizeof(__half), cudaMemcpyDeviceToHost));  

    // __global__ void check(size_t batchCount, size_t m , size_t n , __half* d_A, __half* d_x, __half* d_y, __half* d_y_tc)
    dim3 check_block = 1;
    dim3 check_threads = 1;
    // check_kernel<<<check_block, check_threads>>>(batchCount,m,n,d_A,d_x,d_y_test,d_y);

    cudaDeviceSynchronize();


    // for( int b = 0; b < batchCount; b++)
    // {
    //   for( int i = 0; i < m; i++)
    //   {
    //     __half tmp = 0.0f;
    //     for(int j = 0; j < n; j++)
    //     {
    //       tmp += h_A[b * m * n + i * n + j] * h_x[b * n + j];
    //     }
    //     h_y_blas[b * m + i] = tmp;
    //     printf("%f \n", f16_to_f32_cpu(tmp));
    //   }
    // }

    // for(int i = 0; i < m * batchCount; i++)
    // {
    //   if(fabs(f16_to_f32_cpu(h_y[i]) - f16_to_f32_cpu(h_y_blas[i])) > 1e-5)
    //   {
    //     printf("ERROR: pos : %d blas: %f , my: %f \n " , i, f16_to_f32_cpu(h_y_blas[i]), f16_to_f32_cpu(h_y[i]));
    //   }
    // }


    
    // for (size_t i = 0; i < m * batchCount; i++)
    // {
    //     /* code */
    //    float tmp = f16_to_f32_cpu(h_y[i]);
    //    printf("%f \n",tmp);
    //    if(i%MMA_M==0&&i!=0) printf("\n");
    //    if(i%m==0&&i!=0) printf("\n");
    // }

    return EXIT_SUCCESS;  
}
