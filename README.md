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
	* Arithmetic latency: the time between an arithmetic operation starting and its output being produced. (10-20 cycles)
	* Memory instructions: the time between a load or store operation being issued and the data arriving its destination. (400-800 cycles)
* Use *Little's Law* to estimate the number of active warps required to hide latency:
	* num_warps_needed_to_hide_latency = avg_instruction_latency * through_output_of_warps_per_cycle
* Two ways to increase parallelism:
	* *Instruction-level parallelism (ILP)*: Most independent instructions within a thread
	* *Thread-level parallelism (TLP)*: More concurrently eligible threads
* Choosing an optimal execution configuration is a matter of striking a balance between latency hiding and resource utilization.

#### Occupancy
* *Occupancy* is the ratio of active warps to maximum number of warps, per SM.
* Use *CUDA Occupancy Calculator* to select grid and block dimensions to maximize occupancy for a kernel.
* Manipulating thread blocks to either extreme can restrict resource utilization:
	* Small thread blocks: too few threads per block leads to hardware limits on the number of warps per SM to be reached before all resources are fully utilized.
	* Large thread blocks: too many threads per block leads to fewer per-SM hardware resources available to each thread.
* Guidelines for Grid and Block Size:
	* Keep the number of threads per block a multiple of warp size (32).
	* Avoid small block sizes: Start with at least 128 or 256 threads per block.
	* Adjust block size up or down according to kernel resource requirements.
	* Keep the number of blocks much greater than the number of SMs to expose sufficient parallelism to your device.
	* Conduct experiments to discover the best execution configuration and resource usage.

#### Synchronization

* Synchronization can be performed at two levels:
	* *System-level*: wait for all work on both the host and the device to complete.
	* *Block-level*: wait for all threads in a thread block to reach the same point in execution on the device.
* **cudaDeviceSynchronize()** wait for all operations (copies, kernels, and so on) have completed, also it returns errors from previous asynchronous operations.
* Mark synchronization points in the kernel using **__device__ void __syncthreads(void);**.
* When **__syncthreads** is called, each thread in the same thread block must wait until all other threads in that thread block have reached this synchronization point. All global and shared memory accesses made by all threads prior to this barrier will be visible to all other threads in the thread block after the barrier.
* The only safe way to synchronize across block is to use the global synchronization point at the end of every kernel execution (and then maybe start a new one).
* Not allowing threads in different blocks to synchronize with each other, GPUs can execute blocks in any order. This enables CUDA programs to be scalable across massively parallel GPUs.

#### Scalability

* Real scalability depends on algorithm design and hardware features.
* The ability to execute the same application code on a varying number of compute cores is referred to as *transparent scalability*.
* Scalability can be more important than efficiency.
	* A scalable but inefficient system can handle larger workloads by simply adding hardware cores.
	* An efficient but un-scalable system may quickly reach an upper limit on achievable performance.

### Exposing Parallelism

* *Achieved occupancy*: avg_active_warps_per_cycle / max_num_waprs_supported_on_sm
* *Global load efficiency*: requested_global_load_throughput / required_global_load_throughput

```bash
$ nvprof --metrics achieved_occupancy ./runnable_name // [0, 1]
$ nvprof --metrics gld_throughput ./runnable_name // GB/s
$ nvprof --metrics gld_efficiency ./runnable_name // [percentage]% (global load efficiency)
```

* Innermost dimension should always be a multiple of the warp size (regardless of the other two dimensions).
* Metrics and performance:
	* In most cases, no single metric can prescribe optimal performance.
	* Which metric or event most directly relates to overall performance depends on the nature of the kernel code.
	* Seek a good balance among related metrics and events.
	* Check the kernel from different angles to find a balance among the related metrics.
	* Grid/block heuristics provide a good starting point for performance tuning.

### Avoiding Branch Divergence

#### The Parallel Reduction Problem

* Two types of pair:
	* *Neighbored pair*: Elements are paired with their immediate neighbor.
	* *Interleaved pair*: Paired elements are separated by a given stride.

```cuda
// Finished implementation of computing the sum of array of size n:

int HostFunc(int* const h_odata,
			 int* const d_idata,
			 int* const d_odata,
			 std::size_t const n)
{
	// The following kernel launch configuration is the most optimal one on Titan X (Pascal).
	// Which reached gld_throughput 223GB/s. This configuration may not work as well on
	// other hardwares.
	dim3 const block_dim {64};
	dim3 const grid_dim {(n + block_dim.x - 1)/block_dim.x};
	unsigned short const unrolling_factor {8};
	ReduceUnrolling8<int, 64><<<grid_dim.x/unrolling_factor, block_dim>>>(d_idata, d_odata, n);

	// Use CPU to calculate the remaining sum
	CheckCUDAErr(cudaMemcpy(h_odata, d_odata,
							grid_dim.x/unrolling_factor * sizeof(int),
							cudaMemcpyDeviceToHost));
	int gpu_sum {0};
	for (std::size_t i {0}; i < grid_dim.x/unrolling_factor; ++i) {
		gpu_sum += h_odata[i];
	}
	return gpu_sum;
}

template<typename Dtype, unsigned int i_block_size>
__global__ void ReduceUnrolling8(Dtype* const g_idata, Dtype* const g_odata, std::size_t const n) {
    // Overall index of this element
    unsigned int const tid {threadIdx.x};
    unsigned int const idx {tid + blockIdx.x * 8 * i_block_size};
    unsigned int const block_idx_x {blockIdx.x};

    // Get pointer for the first element of this block
    Dtype* const idata {g_idata + block_idx_x * i_block_size * 8};

    // Because blocks are scalable across multiple SMs, instead of unrolling data inside one block, we unroll data
    // across multiple blocks (block-level parallel).
    // Unrolling 8 data blocks
    bool const ipred {idx + 7 * i_block_size < n};
    if (ipred) {
    	// WARNING: Intentional unrolling, do NOT convert to for-loop.
    	Dtype const a0 {g_idata[idx]};
    	Dtype const a1 {g_idata[idx + i_block_size]};
    	Dtype const a2 {g_idata[idx + i_block_size * 2]};
    	Dtype const a3 {g_idata[idx + i_block_size * 3]};
    	Dtype const a4 {g_idata[idx + i_block_size * 4]};
    	Dtype const a5 {g_idata[idx + i_block_size * 5]};
    	Dtype const a6 {g_idata[idx + i_block_size * 6]};
    	Dtype const a7 {g_idata[idx + i_block_size * 7]};
    	g_idata[idx] = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
    	__syncthreads();
    }

    if (i_block_size >= 1024 && tid < 512) {
    	idata[tid] += idata[tid + 512];
    	__syncthreads();
    }

    if (i_block_size >= 512 && tid < 256) {
		idata[tid] += idata[tid + 256];
		__syncthreads();
	}

    if (i_block_size >= 256 && tid < 128) {
		idata[tid] += idata[tid + 128];
		__syncthreads();
	}

    if (i_block_size >= 128 && tid < 64) {
		idata[tid] += idata[tid + 64];
		__syncthreads();
	}

	if (tid < 32) {
		volatile Dtype *vmem {idata};
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	// After all the reduction completed, first thread of block will copy the number to
	// device output data (with size n/blockDim.x);
	if (tid == 0) {
		g_odata[block_idx_x] = idata[0];
	}
}

/* Sample output:
Array size and type: int[1 << 24]
Kernel Launch Configuration: <<<(262144, 1, 1), (64, 1, 1)>>>
CPU sum 			  :  16.59ms 		  (2139353472.00)
Reduce Neighbored     :  13.53ms  14.21ms (2139353472.00)
Reduce Neighbored Less:   7.32ms   7.64ms (2139353472.00)
Reduce Interleaved    :  11.47ms  11.81ms (2139353472.00)
Reduce Unrolling 2    :   6.15ms   6.37ms (2139353472.00)
Reduce Unrolling 8    :   0.76ms   0.81ms (2139353472.00)
*/
```
