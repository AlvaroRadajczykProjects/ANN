#include "HostPinnedDeviceMatrix.cuh"

HostPinnedDeviceMatrix::HostPinnedDeviceMatrix(int nr, int nc, int mo, unsigned int flags) {
	number_rows = nr;
	number_cols = nc;
	total_accesible_elements = number_rows * number_cols;
	number_elements_multiple_of = mo;
	total_elements = nextMultiple(total_accesible_elements, number_elements_multiple_of);

	cudaHostAlloc(&host_pinned_data, total_elements * sizeof(float), flags);
	cudaMalloc(&device_data, total_elements * sizeof(float));

	cudaDeviceSynchronize();
}

HostPinnedDeviceMatrix::~HostPinnedDeviceMatrix() {
	number_rows = 0;
	number_cols = 0;
	total_accesible_elements = 0;
	number_elements_multiple_of = 1;
	total_elements = 0;

	cudaFreeHost(host_pinned_data);
	cudaFree(device_data);

	cudaDeviceSynchronize();
}

int HostPinnedDeviceMatrix::getNumberRows() {
	return number_rows;
}

int HostPinnedDeviceMatrix::getNumberCols() {
	return number_cols;
}

float* HostPinnedDeviceMatrix::getDevicePointer() {
	return device_data;
}

void HostPinnedDeviceMatrix::copyHostToDevice(float* h_ptr, cudaStream_t stream) {
	manageCUDAError(cudaMemcpyAsync(host_pinned_data, h_ptr, total_accesible_elements * sizeof(float), cudaMemcpyHostToHost, stream), "HostPinnedDeviceMatrix copyHostToDevice: 1");
	manageCUDAError(cudaMemcpyAsync(device_data, host_pinned_data, total_accesible_elements * sizeof(float), cudaMemcpyHostToDevice, stream), "HostPinnedDeviceMatrix copyHostToDevice: 2");
}

void HostPinnedDeviceMatrix::copyDeviceToHost(float* h_ptr, cudaStream_t stream) {
	manageCUDAError(cudaMemcpyAsync(device_data, host_pinned_data, total_accesible_elements * sizeof(float), cudaMemcpyDeviceToHost, stream), "HostPinnedDeviceMatrix copyDeviceToHost: 1");
	manageCUDAError(cudaMemcpyAsync(h_ptr, host_pinned_data, total_accesible_elements * sizeof(float), cudaMemcpyHostToHost, stream), "HostPinnedDeviceMatrix copyDeviceToHost: 2");
}

void HostPinnedDeviceMatrix::showDeviceMatrix(char* msg, cudaStream_t stream) {
	float* mtr = new float[total_accesible_elements];
	copyDeviceToHost(mtr, stream);
	cudaStreamSynchronize(stream);
	imprimirMatrizPorPantalla(msg, mtr, number_rows, number_cols);
	delete mtr;
}