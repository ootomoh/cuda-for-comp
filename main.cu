#include <iostream>
#include <chrono>

__global__ void dev_kernel_A(const int N,float *ptr){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= N) return;

	for(int i = 0;i < N;i++){
		ptr[i] = ptr[i] * (-1.0f);
	}
}
__global__ void dev_kernel_B(const int N,float *ptr){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= N/2) return;

	for(int i = 0;i < N/2;i++){
		ptr[2 * i    ] = ptr[2 * i    ] * (-1.0f);
		ptr[2 * i + 1] = ptr[2 * i + 1] * (-1.0f);
	}
}


template<int N>
__global__ void dev_kernel_C(float *ptr){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= N) return;

	for(int i = 0;i < N;i++){
		ptr[i] = ptr[i] * (-1.0f);
	}
}
template<int N>
__global__ void dev_kernel_D(float *ptr){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= N/2) return;

	for(int i = 0;i < N/2;i++){
		ptr[2 * i    ] = ptr[2 * i    ] * (-1.0f);
		ptr[2 * i + 1] = ptr[2 * i + 1] * (-1.0f);
	}
}



template<int N>
__global__ void dev_kernel_E(float *ptr){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= N) return;

#pragma unroll
	for(int i = 0;i < N;i++){
		ptr[i] = ptr[i] * (-1.0f);
	}
}
template<int N>
__global__ void dev_kernel_F(float *ptr){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= N/2) return;

#pragma unroll
	for(int i = 0;i < N/2;i++){
		ptr[2 * i    ] = ptr[2 * i    ] * (-1.0f);
		ptr[2 * i + 1] = ptr[2 * i + 1] * (-1.0f);
	}
}

template<class FuncPre,class FuncRun,class FuncFin>
void printElapsedTime(FuncPre pre,FuncRun run,FuncFin fin){
	pre();
	auto start = std::chrono::system_clock::now();
	run();
	auto end = std::chrono::system_clock::now();
	std::cout<<"Elapsed time : "<<std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()<<" [us]"<<std::endl;
	fin();
}

int main(){
	const int N = 1000;
	const int C = 1000;
	const int ROUND = 3;
	const int num_threads = 32;
	float *dptr;

	for(int r = 0;r < ROUND;r++){
		std::cout<<"##-- "<<"ROUND "<<(r+1)<<std::endl;
		std::cout<<"#-- simple for"<<std::endl;
		printElapsedTime(
				[&dptr](){
				cudaMalloc((void**)&dptr,sizeof(float) * N);
				printf("ptr : %p\n",dptr);
				},
				[&dptr](){
				for(int i = 0;i < C;i++)
				dev_kernel_A<<<(N+num_threads-1)/num_threads,num_threads>>>(N,dptr);
				cudaDeviceSynchronize();
				},
				[&dptr](){cudaFree(dptr);}
				);
		std::cout<<"#-- divided by 2 for"<<std::endl;
		printElapsedTime(
				[&dptr](){
				cudaMalloc((void**)&dptr,sizeof(float) * N);
				printf("ptr : %p\n",dptr);
				},
				[&dptr](){
				for(int i = 0;i < C;i++)
				dev_kernel_B<<<(N+num_threads-1)/num_threads,num_threads>>>(N,dptr);
				cudaDeviceSynchronize();
				},
				[&dptr](){cudaFree(dptr);}
				);
		std::cout<<"#-- N assembled, simple for"<<std::endl;
		printElapsedTime(
				[&dptr](){
				cudaMalloc((void**)&dptr,sizeof(float) * N);
				printf("ptr : %p\n",dptr);
				},
				[&dptr](){
				for(int i = 0;i < C;i++)
				dev_kernel_C<N><<<(N+num_threads-1)/num_threads,num_threads>>>(dptr);
				cudaDeviceSynchronize();
				},
				[&dptr](){cudaFree(dptr);}
				);
		std::cout<<"#-- N assembled, divided by 2 for"<<std::endl;
		printElapsedTime(
				[&dptr](){
				cudaMalloc((void**)&dptr,sizeof(float) * N);
				printf("ptr : %p\n",dptr);
				},
				[&dptr](){
				for(int i = 0;i < C;i++)
				dev_kernel_D<N><<<(N+num_threads-1)/num_threads,num_threads>>>(dptr);
				cudaDeviceSynchronize();
				},
				[&dptr](){cudaFree(dptr);}
				);
		std::cout<<"#-- N assembled, simple, for (expressly unrolled)"<<std::endl;
		printElapsedTime(
				[&dptr](){
				cudaMalloc((void**)&dptr,sizeof(float) * N);
				printf("ptr : %p\n",dptr);
				},
				[&dptr](){
				for(int i = 0;i < C;i++)
				dev_kernel_C<N><<<(N+num_threads-1)/num_threads,num_threads>>>(dptr);
				cudaDeviceSynchronize();
				},
				[&dptr](){cudaFree(dptr);}
				);
		std::cout<<"#-- N assembled, divided by 2, for (expressly unrolled)"<<std::endl;
		printElapsedTime(
				[&dptr](){
				cudaMalloc((void**)&dptr,sizeof(float) * N);
				printf("ptr : %p\n",dptr);
				},
				[&dptr](){
				for(int i = 0;i < C;i++)
				dev_kernel_D<N><<<(N+num_threads-1)/num_threads,num_threads>>>(dptr);
				cudaDeviceSynchronize();
				},
				[&dptr](){cudaFree(dptr);}
				);
		std::cout<<std::endl;
	}
}
