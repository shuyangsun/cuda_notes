# CUDA Notes

___

## 1. Introduction
* A *data dependency* occurs when an instruction consumes data produced by a preceding instruction.
* Two fundamental types of parallelism:
	1. Task parallelism: when there are many data items that can be operated on at the same time. Focuses on distributing functions across multiple cores.
	2. Data parallelism (what CUDA focuses on): when there are many data items that can be operated on at the same time. Focuses on distributing the data across multiple cores.

```cuda
// A simple CUDA program
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
```
