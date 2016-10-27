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

### "Hello World" from GPU

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

___

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

#### Grids and Blocks

* All threads spawned by a single kernel launch are collectively called a *grid*.
* All threads in a grid share the same global memory space.
* A grid is made up of many thread blocks.
* A thread block is a group of threads that can cooperate with each other using:
	* Block-local synchronization
	* Block-local shared memory
* Threads from different blocks cannot cooperate.

#### Threads Organization

* Threads rely on the following two unique coordinates to distinguish themselves from each other:
	* **blockIdx**: block index within a grid
	* **threadIdx**: thread index within a block
* These variables are built-in, pre-initialized variables that can be accessed within kernel functions.

* The coordinate variable is of type **uint3**, a CUDA built-in vector type. The three elements can be accessed through **x**, **y**, and **z** respectively.

```cuda
// Block index
typeof(blockIdx); // uint3
unsigned int bId_x {blockIdx.x};
unsigned int bId_y {blockIdx.y};
unsigned int bId_z {blockIdx.z};

// Thread index
typeof(threadIdx); // uint3
unsigned int tId_x {threadIdx.x};
unsigned int tId_y {threadIdx.y};
unsigned int tId_z {threadIdx.z};
```

#### Grids and Blocks Organization

* CUDA organizes grids and blocks in three dimensions. The dimensions of a grid and a block are specified by the following two built-in variables:
	* **blockDim**: block dimension, measured in threads
	* **gridDim**: grid dimension, measured in blocks
* These variables are of type **dim3**. Use **x**, **y**, and **z** to access respectively.
* There are two distinct sets of grid and block variables in a CUDA program: manually-defined **dim3** data type and pre-defined **uint3** data type:
	* On the host side, you define the dimensions of a grid and block using **dim3** data type as part of a kernel invocation.
	* When the kernel is executing, the CUDA runtime generates the corresponding built-in, pre-initialized grid, block, and thread variables, which are accessible within the kernel function and have type **uint3**.
	* The manually created **dim3** variables are only visible on the host side; the built-in **uint3** variables are only visible on the device side.

| Host           | Device              |
| -------------- | ------------------- |
| **dim3** grid  | **uint3** gridDim   |
| **dim3** block | **uint3** blockDim  |
|                | **uint3** blockIdx  |
|                | **uint3** threadIdx |

* Host side: manually create; type of **dim3**
* Device side: built-in, pre-initialized; type of **uint3**

```cuda
const dim3 block {3};
const dim3 grid {(num_ele + block.x - 1) / block.x};

check_index_host(grid, block);
check_index_device<<<grid, block>>>();

cudaDeviceReset();

/* Output:
-------- HOST -------
grid (2, 1, 1) block (3, 1, 1)
---------------------

------- DEVICE ------
threadIdx (0, 0, 0) blockIdx (0, 0, 0) blockDim (3, 1, 1) gridDim (2, 1, 1)
threadIdx (1, 0, 0) blockIdx (0, 0, 0) blockDim (3, 1, 1) gridDim (2, 1, 1)
threadIdx (2, 0, 0) blockIdx (0, 0, 0) blockDim (3, 1, 1) gridDim (2, 1, 1)
threadIdx (0, 0, 0) blockIdx (1, 0, 0) blockDim (3, 1, 1) gridDim (2, 1, 1)
threadIdx (1, 0, 0) blockIdx (1, 0, 0) blockDim (3, 1, 1) gridDim (2, 1, 1)
threadIdx (2, 0, 0) blockIdx (1, 0, 0) blockDim (3, 1, 1) gridDim (2, 1, 1)
---------------------
*/
```
* For a given data size, the general steps to determine the grid and block dimensions are:
	* Decide the block size.
	* Calculate the grid dimension based on the application data size and the block size.
* To determine the block dimension, you usually need to consider:
	* Performance characteristics of the kernel
	* Limitations on GPU resources

### Launching a CUDA Kernel

* Kernel calls are asynchronous.
* call **cudaDeviceSynchronize()** to wait for all kernels to complete.
* **cudaMemcpy** implicitly synchronizes at the host side, it starts to copy after all previous kernel calls have completed.
* A kernel function is the code to be executed on the device side in parallel.
* In a kernel function, you define the computation for a single thread, and the data access for that thread.
* A kernel function is defined using **__global__** declaration specification, and must have a **void** return type.

```cuda
__global__ void kernel_name(/* argument list */);
```
| Qualifiers | Execution | Callable | Notes |
| --- | --- | --- | --- |
| **\__global__** | Executed on the device | Callable from the host <br /> Callable from the device for devices of compute capability 3 | Must have a **void** return type |
| **\__device__** | Executed on the device | Callable from the device only | |
| **\__host__** | Executed on the host | Callable from the host only | Can be omitted |

* The **\__device__** and **\__host__** qualifiers can be used together, in which case the function is compiled for both the host and the device.
* Restrictions for CUDA kernel functions:
	* Access to device memory only
	* Must have **void** return type
	* No support for a variable number of arguments
	* No support for **static** variables
	* No support for function pointers
	* Exhibit an asynchronous behavior

### Handling Errors

* Use an **inline** function to check **cudaError_t**s returned by CUDA API calls.

```cuda
__host__ inline void check_err(const std::initializer_list<const cudaError_t>& errors) {
	#pragma unroll
	for (auto err: errors) {
		if (err != cudaSuccess) {
			fprintf(stderr, "%s\n", cudaGetErrorString(err));
			throw customized_cuda_exception {"Info about error..."};
		}
	}
}
```

### Timing Your Kernel

#### Timing with CPU Timer

```cuda
// How to get device information
__host__ inline void print_device_info(void) {
	unsigned int device_idx {0};
	cudaDeviceProp device_prop {};
	cudaGetDeviceProperties(&device_prop, device_idx);
	std::cout << "Using device " << device_idx << ": " << device_prop.name << " ";
	std::cout << "with " << device_prop.multiProcessorCount << " SM units" << std::endl;
}
```

#### Timing with nvprof

* Timing with **nvprof** is more accurate than using CPU timer.

```bash
$ nvprof ./runnable_name

# Sample output:
# 
# ==10911== Profiling application: ./arrsum
# ==10911== Profiling result:
# Time(%)      Time     Calls       Avg       Min       Max  Name
#  62.90%  362.41ms         2  181.20ms  181.09ms  181.32ms  [CUDA memcpy HtoD]
#  35.52%  204.69ms         1  204.69ms  204.69ms  204.69ms  [CUDA memcpy DtoH]
#   1.58%  9.1138ms         1  9.1138ms  9.1138ms  9.1138ms  sum_arr_gpu(float const *, float const *, float*, unsigned int)
# 
# ==10911== API calls:
# Time(%)      Time     Calls       Avg       Min       Max  Name
#  59.22%  567.27ms         3  189.09ms  180.96ms  204.89ms  cudaMemcpy
#  23.42%  224.37ms         3  74.792ms  474.22us  223.42ms  cudaMalloc
# 
#  ... etc.

```

* For HPC workloads, it is important to understand the compute to communication ratio in a program.
* It's important to determine how your application compares to theoretical limits.

## 3. CUDA Execution Model

### GPU Architecture Overview

* The GPU architecture is built around a scalable array of *Streaming Multiprocessors* (SM).
* When a kernel grid is launched, the thread blocks of that kernel grid are distributed among available SMs for execution. Once scheduled on an SM, the threads of a thread block execute concurrently only on that assigned SM. Multiple thread blocks may be assigned to the same SM at once and are scheduled based on the availability of SM resources. Instructions within a single thread are pipelined to leverage instruction-level parallelism, in addition to the thread-level parallelism you are already familiar within CUDA.
* CUDA employs a *Single Instruction Multiple Thread* (SIMT) architecture to manage and execute threads in groups of 32 called *warps*. All threads in a warp execute the same instruction at the same time. Each thread has its own instruction address counter and register state, and carries out the current instruction on its own data. Each SM partitions the thread blocks assigned to it into 32-thread warps that it then schedules for execution on available hardware resources.
* A key difference between SIMD and SIMT is that, SIMD requires that all vector elements in a vector execute together in a unified synchronous group, whereas SIMT allows multiple threads in the same warp to execute independently. Even though all threads in a warp start together at the same program address, it is possible for individual threads to have different behavior.
* The SIMT model includes three key features that SIMD does not:
	* Each thread has its own instruction address counter.
	* Each thread has its own register state.
	* Each thread can have an independent execution path.
* The number 32 is a magic number in CUDA programming. It comes from hardware, and has a significant impact on the performance of software.
* A thread block is scheduled on only one SM. Once a thread block is scheduled on an SM, it remains there until execution completes. An SM can hold more than one thread block at the same time.
* Shared memory and registers are precious resources in an SM. Shared memory is partitioned among thread blocks resident on the SM and registers are partitioned among threads. Threads in a thread block can cooperate and communicate with each other through these resources. While all threads in a thread block run logically in parallel, not all threads can execute physically at the same time. As a result, different threads in a thread block may make progress at a different pace.
* Threads in a block can communicate with each other, but no primitives are provided for inter-block synchronization.
* Switching between concurrent warps has no overhead because hardware resources are partitioned among all threads and blocks on an SM.

#### The Fermi Architecture:

* Some terminologies:
	* LD/ST: load/store unit
	* ALU: arithmetic logic unit
	* FPU: floating-point unit (executes one integer or floating point instruction per clock cycle)
	* SFU: special function unit (sin, cos, etc.)

![alt text][fermi_arch]
[fermi_arch]: resources/Fermi_architecture.png "Fermi Architecture"
(from https://commons.wikimedia.org/wiki/File:Fermi.svg)

![alt text][fermi_drawing_01]
[fermi_drawing_01]: resources/Fermi_drawing.jpg "Fermi Processing Flow"

#### The Kepler Architecture

* New features in Kepler:
	* Enhanced SMs
	* Dynamic Parallelism: launch another kernel within a kernel
	* Hyper-Q: enabling CPU cores to simultaneously run more tasks on the GPU.

![alt text][kepler_arch_01]
[kepler_arch_01]: resources/kepler_01.jpg "Kepler Architecture"
(from http://electronicdesign.com/site-files/electronicdesign.com/files/archive/electronicdesign.com/content/content/74002/74002_fig2.jpg)

![alt text][kepler_arch_02]
[kepler_arch_02]: resources/kepler_02.jpg "SMX vs. SM"
(from http://images.bit-tech.net/content_images/2012/03/nvidia-geforce-gtx-680-2gb-review/gtx680-21b.jpg)


### Profile-Driven Optimization

* Profiling is the act of analyzing program performance by measuring:
	* The space (memory) or time complexity of application code
	* The use of particular instructions
	* The frequency and duration of function calls
* CUDA provides two primary profiling tools:
	* **nvvp**: a standalone visual profiler
	* **nvprof**: a command-line profiler
* Three common limiters to performance for a kernel:
	* Memory bandwidth
	* Compute resources
	* Instruction and memory latency

### Understanding the Nature of Warp Execution

#### Warps and Thread Blocks

* Threads with consecutive values for **threadIdx.x** are grouped into warps.

```cuda
// Example: 1D thread block with 64 threads
// Warp 0: thread  0, thread  1, thread  2, ..., thread 32
// Warp 1: thread 32, thread 33, thread 34, ..., thread 63

// For 2 or 3 dimensional thread block it's the same, just convert the index to 1D
const int idx_1d = threadIdx.x;
const int idx_2d = (threadIdx.y * threadIdx.x) + threadIdx.x;
const int idx_3d = (threadIdx.z * threadIdx.y * threadIdx.x) + (threadIdx.y * threadIdx.x) + threadIdx.x;
```

* warps_per_block = ceil(threads_per_block / warp_size)
* If thread block size is not an even multiple of warp size, some threads in the last warp are left inactive.
* No matter what the block and grid dimension is, from the hardware perspective, a thread block is a 1D collection of warps.

#### Warp Divergence

```cuda
// Example
__global__ void foo() {
	if (condition) {
		// some threads will execute this block of code
	} else {
		// some threads will execute this one instead
	}
}
// contradict with the rule that all threads must run the same code
```

* Threads in the same warp executing different instructions is referred to as *warp divergence*.
* If threads of a warp diverge, the warp serially executes each branch path, disabling threads that do not take that path.
* Warp divergence can cause significantly degraded performance.
* Different condition values in different warps do not cause warp divergence.
* It may be possible to partition data in such a way as to ensure all threads in the same warp take the same control path in an application.
* branch_efficiency = (num_branches - num_divergent_branches) / num_branches
* Some times warp divergence does not happen (when it should), that's because CUDA compiler optimization. It replaces branch instructions (which cause actual control flow to diverge) with predicated instructions for short, conditional code segments.

##### Branch Prediction
* In branch prediction, a predicate variable for each thread is set to 1 or 0 according to a conditional. Both conditional flow paths are fully executed, but only instructions with a predicate of 1 are executed. Instructions with a predicate of 0 do not, but the corresponding thread does not stall either. The difference between this and actual branch instructions is subtle, but important to understand. The compiler replaces a branch instruction with predicted instructions only if the number of instructions in the body of a conditional statement is less than a certain threshold. Therefore, a long code path will certainly result in warp divergence.

```bash
$ nvprof --metrics branch_efficiency ./runnable-name # Check each kernel's warp divergence efficiency
$ nvcc -g -G ./runnable-name # Disable kernel optimization
$ nvprof --events branch,divergent_branch ./runnable-name # Check branch counter
```
#### Resource Partitioning

* The local execution context of a warp mainly consists of the following resources:
	* Program counters
	* Registers
	* Shared memory

* The execution context of each warp processed by an SM is maintained on-chip during the entire lifetime of the warp. Therefore, switching from one execution context to another has no cost.
* Changing the number of registers and the amount of shared memory required by the kernel, can change the number of blocks and warps that can simultaneously reside on an SM.
* If there are insufficient registers or shared memory on each SM to process at least one block, the kernel launch will fail.
* A thread block is called an *active block* when compute resources such as registers and shared memory, have been allocated to it. The warps it contains are called *active warps*. Active warps can be further classified into the following three types:
	* Selected warp: actively executing
	* Stalled warp: ready for execution but not currently executing
	* Eligible warp: not ready for execution
* The warp scheduler on an SM select active warps on every cycle and dispatch them to execution units.
* A warp is eligible for execution if both of the following two conditions are met:
	* 32 CUDA cores are available for execution.
	* All arguments to the current instruction are ready.
* On Kepler SM, the number of active warps is limited to 64, and the number of selected warps at any cycle is less than or equal to 4.
* The compute resources limit the number of active warps. Therefore, you must be aware of the restrictions imposed by the hardware, and the resources used by your kernel. In order to maximize GPU utilization, you need to maximize the number of active warps.

#### Latency Hiding

* The number of clock cycles between an instruction being issued and being completed is defined as *instruction latency*.
* Full compute resource utilization is achieved when all warp schedulers have an eligible warp at every clock cycle. This ensures the latency of each instruction can be hidden by issuing other instructions in other resident warps.
* Two types of instructions (which could have latency):
	* Arithmetic instructions
	* Memory instructions
* Two types of latency:
	* Arithmetic latency: the time between an arithmetic operation starting and its output being produced.
	* Memory instructions: the time between a load or store operation being issued and the data arriving its destination.


