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
* **Host code** runs on CPUs and **device code** runs on GPUs.
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

__global__ void helloFromGPU() {
	printf("Hello from GPU!\n");
}


int main(int argc, const char* argv[]) {

	// Triple angle brackets mark a call from the host thread to the code on the device side.
	// A kernel's executed by an array of threads and all threads run the same code.

	helloFromGPU<<<1, 5>>>(); // Call helloFromGPU 5 times on GPU

	// Explicitly destroy and clean up all resources associated with the current device in the current process.
	cudaDeviceReset();

	return 0;
}

// Compile with "nvcc -arch sm_61 hello.cu -o hello"
// "-arch sm_<computing capability>" is to compile for a specific architecture.
```
