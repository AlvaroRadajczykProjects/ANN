#include "CUDAKernels.cuh"
#include "PrintUtils.h"

#ifndef HOST_PINNED_DEVICE_MATRIX
#define HOST_PINNED_DEVICE_MATRIX

class HostPinnedDeviceMatrix {

	private:
		//number of accesible matrix rows
		int number_rows;
		//number of accesible matrix cols
		int number_cols;
		//number of total accesible elements in matrix
		unsigned long long total_accesible_elements;
		//number of total (accesible and not accesible) elements in matrix
		unsigned long long total_elements;
		//max common divisor for the total number of elements
		int number_elements_multiple_of = 1;
		//host pinned matrix
		float* host_pinned_data = NULL;
		//device matrix
		float* device_data = NULL;

	public:
		HostPinnedDeviceMatrix(int nr, int nc, int mo = 1, unsigned int flags = 0);
		~HostPinnedDeviceMatrix();

		int getNumberRows();
		int getNumberCols();
		float* getDevicePointer();

		void copyHostToDevice(float* h_ptr, cudaStream_t stream);
		void copyDeviceToHost(float* h_ptr, cudaStream_t stream);

		void showDeviceMatrix(char* msg, cudaStream_t stream);

};

#endif