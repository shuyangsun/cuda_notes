# CUDA Notes

___

## 1. Introduction

```cuda
// A simple cuda program
#include <cstdio>

__global__ void helloFromGPU() {
	printf("Hello from GPU!\n");
}


int main(int argc, const char* argv[]) {

	// Tripple angle brackets mark a call from the host thread to the code on the device side.
	// A kernel's executed by an array of threads and all threads run the same code.
	helloFromGPU<<<1, 5>>>();

	// Explicitly destroy and clean up all resources associated with the current device in the current process.
	cudaDeviceReset();

	return 0;
}
```
