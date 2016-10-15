# CUDA Notes

___

## 1. Introduction

### Sequential and Parallel Programming

* A *data dependency* occurs when an instruction consumes data produced by a preceding instruction.

### Parallelism

* Two fundamental types of parallelism:
	1. Task parallelism: when there are many data items that can be operated on at the same time. Focuses on distributing functions across multiple cores.
	2. Data parallelism (what CUDA focuses on): when there are many data items that can be operated on at the same time. Focuses on distributing the data across multiple cores.

* Even when a logical multi-dimensional view of data is used, it still maps to one-dimensional physical storage.
* Two ways of partitioning memory: *block partition* and *cyclic partition*.
* The way you organize threads has a significant effect on the program's performance.

### Computer Architecture

* Four main architectures:
	* Single Instruction Single Data (SISD)
	* Single Instruction Multiple Data (SIMD)
	* Multiple Instruction Single Data (MISD)
	* Multiple Instruction Multiple Data (MIMD)

* On architectural level, they are trying to: decrease latency, increase bandwidth, increase throughput
	* *Latency*: the time it takes for an operation to start and complete, and is commonly expressed in microseconds.
	* *Bandwidth*: the amount of data that can be processed per unit of time, commonly expressed as megabytes/sec or gigabytes/sec.
	* *Throughput*: the amount of operations that can be processed per unit of time, commonly expressed as *gflops*.

* A multi-node system is often referred to as *clusters*.
* Single Instruction Multiple Thread (**SIMT**): NVIDIA's term for the combination of all four main architectures.

### Heterogeneous Architecture

* Each hardware performs tasks it does the best.
* *Host code* runs on CPUs and *device code* runs on GPUs.
* GPU is a type of *hardware accelerator*.

### Paradigm of Heterogeneous Computing

* CPU is preferred for small data size, low level of parallelism computing.
* GPU is preferred for large data size, high level of parallelism computing.
* GPU threads are extremely lightweight, context switch is almost instantaneous; in contrast, CPU threads are heavyweight, context switch is slow and expensive.

### CUDA: A Platform for Heterogeneous Computing

* Runtime API vs. Driver API: no noticeable performance difference, cannot mix together.
* The device code is written using CUDA C extended with keywords for labeling data-parallel functions, called **kernels**.
* You can create or extend programming languages with support for GPU acceleration using the CUDA Compiler SDK.

### HELLO WORLD FROM GPU

```cuda
#include <cstdio>

__global__ void helloFromGPU(void) {
	char str[] {"Hello from GPU!"};
	printf("%s (thread %d)\n", str, threadIdx.x);
}

int main(int argc, const char* argv[]) {

	// Triple angle brackets mark a call from the host thread to the code on the device side.
	// A kernel's executed by an array of threads and all threads run the same code.

	helloFromGPU<<<1, 5>>>(); // Call helloFromGPU 5 times on GPU

	// Explicitly destroy and clean up all resources associated with the current device in the current process.
	cudaDeviceReset();

	return 0;
}

// Compile with "nvcc -arch sm_61 hello.cu -o hello".
// "-arch sm_<computing capability>" is to compile for a specific architecture.
```
* **CUDA Programing Structure**:
	1. Allocate GPU memories.
	2. Copy data from CPU memory to GPU memory.
	3. Invoke the CUDA kernel to perform program-specific computation.
	4. Copy data back from GPU memory to CPU memory.
	5. Destroy GPU memories.

### Is CUDA Programming Difficult?
* *Locality*: the reuse of data so as to reduce memory access latency.
	* **Temporal locality**: the reused of data/resources within relatively small time durations.
	* **Spatial locality**: the reused of data/resources within relatively close storage locations.
* CUDA exposes both thread and memory hierarchy to programmer. (e.g., *shared memory*: software-managed cache)
* Three key abstractions: a hierarchy of thread groups, a hierarchy of memory groups, and barrier synchronization.

## 2. CUDA Programming Model

### Introducing the CUDA Programming Model

* CUDA special features:
	* Organize threads on the GPU through a hierarchy structure
	* Access memory on the GPU through a hierarchy structure

* View parallel computing on different levels:
	* Domain level
	* Logic level
	* Hardware level

### CUDA Programming Structure

* **Host**: the CPU and its memory (host memory)
* **Device**: the GPU and its memory (device memory)
* A key component of the CUDA programming model is the kernel - the code that runs on the GPU device. As the developer, you can express a kernel as a sequential program. Behind the scenes, CUDA manages scheduling programmer-written kernels on GPU threads.

### Managing Memory

| Standard C Functions | CUDA C Functions |
| -------------------- | ---------------- |
| malloc               | cudaMalloc       |
| memcpy               | cudaMemcpy       |
| memset               | cudaMemset       |
| free                 | cudaFree         |


| cudaMemcpyKind               |
| ---------------------------- |
| cudaMemcpyKindHostToHost     |
| cudaMemcpyKindHostToDevice   |
| cudaMemcpyKindDeviceToHost   |
| cudaMemcpyKindDeviceToDevice |

```cuda
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaFree(void* devPtr);

// Every CUDA call, except kernel launches, returns an error code of an enumerated type cudaError_t
// cudaError_t: cudaSuccess, cudaErrorMemoryAllocation, etc.
char *cudaGetErrorString(cudaError_t error);
```
* In the GPU memory hierarchy, the two most important types of memory are *global memory* and *shared memory*. (Both global and shared memory are on GPU)
	* Global memory is analogous to CPU system memory.
	* Shared memory is similar to the CPU cache, can be directly controlled from a CUDA C kernel.

### Organizing Threads

* When a kernel function is launched from the host side, execution is moved to a device where a large number of threads are generated and each thread executes the statements specified by the kernel function. 

* All threads spawned by a single kernel launch are collectively called a *grid*.
* All threads in a grid share the same global memory space.
* A grid is made up of many thread blocks.
* A thread block is a group of threads that can cooperate with each other using:
	* Block-local synchronization
	* Block-local shared memory
* Threads from different blocks cannot cooperate.

* Threads rely on the following two unique coordinates to distinguish themselves from each other:
	* **blockIdx**: block index within a grid
	* **threadIdx**: thread index within a block
* These variables are built-in, pre-initialized variables that can be accessed within kernel functions.

* The coordinate variable is of type **uint3**, a CUDA built-in vector type. The three elements can be accessed through **x**, **y**, and **z** respectively.

* CUDA organizes grids and blocks in three dimensions. The dimensions of a grid and a block are specified by the following two built-in variables:
	* **blockDim**: block dimension, measured in threads
	* **gridDim**: grid dimension, measured in blocks
* These variables are of type **dim3**. Use **x**, **y**, and **z** to access resepectively.
