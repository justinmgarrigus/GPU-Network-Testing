#include <stdlib.h>
#include <iostream>
#include <fstream> 
#include <ctime> 
#include <chrono> 
#include <math.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// #define VERBOSE 
#define DEC_DOUBLE

#define SMALL_WIDTH_INCREASING_LENGTH 
#define LARGE_WIDTH_INCREASING_LENGTH
#define INCREASING_WIDTH_SMALL_LENGTH
#define INCREASING_WIDTH_LARGE_LENGTH 

#if defined(DEC_FLOAT)
typedef float dec;
#elif defined(DEC_DOUBLE)
typedef double dec;
#endif 

unsigned long long start_time;
unsigned long long time_millis() {
	using namespace std::chrono;
	milliseconds ms = duration_cast<milliseconds>(
		system_clock::now().time_since_epoch()
		);
	return ms.count() - start_time; // How many milliseconds have passed since we started the program. 
}

int max_threads_per_block; 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ __host__ int min_val(int left, int right) {
	if (left < right) return left; 
	return right;
}

__device__ __host__ double sigmoid(double z) {
	return 1.0 / (1.0 + exp(-z));
}

__global__ void feedforward_gpu(int max_threads_per_block, int layer, const size_t* lengths, const size_t* lengths_cu,
		  const size_t*lengths_mu, const dec* biases, const dec* weights,
		  dec* nodes) {
	size_t index = max_threads_per_block * blockIdx.x + lengths_cu[layer] + threadIdx.x;
	if (index < lengths_cu[layer+1]) {
		nodes[index] = biases[index];
		for (size_t i = 0; i < lengths[layer - 1]; i++) {
			size_t row = lengths_cu[layer - 1] + i;
			size_t weight_index = lengths_mu[layer] + max_threads_per_block * blockIdx.x + threadIdx.x + i * lengths[layer];
			nodes[index] += nodes[row] * weights[weight_index];
		}
		nodes[index] = sigmoid(nodes[index]);
	}
}

void feedforward_cpu(int layer, const size_t* lengths, const size_t* lengths_cu,
	const size_t* lengths_mu, const dec* biases, const dec* weights,
	dec* nodes) {
	for (size_t rowi = 0; rowi < lengths[layer]; rowi++) {
		size_t index = lengths_cu[layer] + rowi; 
		nodes[index] = biases[index];
		for (size_t i = 0; i < lengths[layer - 1]; i++) {
			size_t row = lengths_cu[layer - 1] + i;
			size_t weight_index = lengths_mu[layer] + rowi + i * lengths[layer];
			nodes[index] += nodes[row] * weights[weight_index];
		}
		nodes[index] = sigmoid(nodes[index]);
	}
}

double random01() {
	return static_cast <dec> (rand()) / static_cast <dec> (RAND_MAX);
}

void random_values(dec* arr, const int length) {
	for (int i = 0; i < length; i++)
		arr[i] = (dec)(random01() * 2 - 1); 
}

void test(int length, int width, int trials, double* gpu_cpu_times) {
	srand(time_millis());
	std::cout << "Length(" << length << ") width(" << width << ") trials(" << trials << "): "; 
    size_t* lengths = new size_t[length]; 
	for (int i = 0; i < length; i++)
		lengths[i] = width; 

	size_t* lengths_cu = new size_t[length + 1];
	lengths_cu[0] = 0;
	for (int i = 1; i <= length; i++)
		lengths_cu[i] = lengths[i - 1] + lengths_cu[i - 1];

#if defined(VERBOSE) 
	std::cout << "lengths_cu: ";
	for (int i = 0; i < length + 1; i++)
		std::cout << lengths_cu[i] << " ";
	std::cout << std::endl;
#endif 

	size_t* lengths_mu = new size_t[length + 1];
	lengths_mu[0] = 0;
	lengths_mu[1] = 0;
	for (int i = 2; i <= length; i++)
		lengths_mu[i] = lengths[i - 2] * lengths[i - 1] + lengths_mu[i - 1];

#if defined(VERBOSE)
	std::cout << "lengths_mu: ";
	for (int i = 0; i < length + 1; i++)
		std::cout << lengths_mu[i] << " ";
	std::cout << std::endl;
#endif 

	size_t* d_lengths; gpuErrchk(cudaMalloc(&d_lengths, sizeof(size_t) * length));
	gpuErrchk(cudaMemcpy(d_lengths, lengths, sizeof(size_t) * length, cudaMemcpyHostToDevice));
	size_t* d_lengths_cu; gpuErrchk(cudaMalloc(&d_lengths_cu, sizeof(size_t) * (length + 1)));
	gpuErrchk(cudaMemcpy(d_lengths_cu, lengths_cu, sizeof(size_t) * (length + 1), cudaMemcpyHostToDevice));
	size_t* d_lengths_mu; gpuErrchk(cudaMalloc(&d_lengths_mu, sizeof(size_t) * (length + 1)));
	gpuErrchk(cudaMemcpy(d_lengths_mu, lengths_mu, sizeof(size_t) * (length + 1), cudaMemcpyHostToDevice));

	dec* nodes = new dec[lengths_cu[length]];
	dec* d_nodes; gpuErrchk(cudaMalloc(&d_nodes, sizeof(dec) * lengths_cu[length]));

	dec* biases = new dec[lengths_cu[length]];
	random_values(biases, lengths_cu[length]);
	dec* d_biases; gpuErrchk(cudaMalloc(&d_biases, sizeof(dec) * lengths_cu[length]));
	gpuErrchk(cudaMemcpy(d_biases, biases, sizeof(dec) * lengths_cu[length], cudaMemcpyHostToDevice));

	dec* weights = new dec[lengths_mu[length]];
	random_values(weights, lengths_mu[length]);
	dec* d_weights; gpuErrchk(cudaMalloc(&d_weights, sizeof(dec) * lengths_mu[length]));
	gpuErrchk(cudaMemcpy(d_weights, weights, sizeof(dec) * lengths_mu[length], cudaMemcpyHostToDevice));

	double test_start_time = (double)time_millis();
	srand(0); 
	dec* result = new dec[lengths_cu[length]];
	for (int trial = 0; trial < trials; trial++) {
		for (size_t row = 0; row < lengths[0]; row++)
			nodes[row] = (dec)random01();
		gpuErrchk(cudaMemcpy(d_nodes, nodes, sizeof(dec) * lengths[0], cudaMemcpyHostToDevice));

		for (int layer = 1; layer < length; layer++) {
			feedforward_gpu<<<1 + lengths[layer] / (max_threads_per_block + 1), min_val(max_threads_per_block, lengths[layer])>>>(max_threads_per_block, layer, d_lengths, d_lengths_cu, d_lengths_mu, d_biases, d_weights, d_nodes);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		gpuErrchk(cudaMemcpy(result, d_nodes + lengths_cu[length - 1], sizeof(dec) * lengths[length - 1], cudaMemcpyDeviceToHost));

#if defined(VERBOSE)
		std::cout << trial << ": ";
		for (int i = 0; i < lengths[length - 1]; i++)
			std::cout << result[i] << " ";
		std::cout << std::endl;
#endif 
	}
	gpu_cpu_times[0] = ((double)time_millis() - test_start_time) / 1000;
	std::cout << "GPU(" << gpu_cpu_times[0] << "s) ";

	test_start_time = (double)time_millis();
	srand(0);
	for (int trial = 0; trial < trials; trial++) {
		for (int row = 0; row < lengths[0]; row++)
			nodes[row] = (dec)random01();

		for (int layer = 1; layer < length; layer++) {
			feedforward_cpu(layer, lengths, lengths_cu, lengths_mu, biases, weights, nodes);
		}

#if defined(VERBOSE)
		std::cout << trial << ": ";
		for (int i = 0; i < lengths[length - 1]; i++)
			std::cout << nodes[lengths_cu[length - 1] + i] << " ";
		std::cout << std::endl;
#endif 
	}
	gpu_cpu_times[1] = ((double)time_millis() - test_start_time) / 1000;
	std::cout << "CPU(" << gpu_cpu_times[1] << "s)" << std::endl;

	// Verifies the output layers of each final trial are equivalent. 
	for (int i = 0; i < lengths[length - 1]; i++) {
		dec diff = abs(result[i] - nodes[lengths_cu[length - 1] + i]);
		if (diff > 0.0001) {
			std::cerr << "Error: difference between CPU and GPU outputs too high (" << diff << ")";
			exit(-1);
		}
	}

	gpuErrchk(cudaFree(d_lengths));
	gpuErrchk(cudaFree(d_lengths_cu));
	gpuErrchk(cudaFree(d_lengths_mu));
	gpuErrchk(cudaFree(d_nodes));
	gpuErrchk(cudaFree(d_biases));
	gpuErrchk(cudaFree(d_weights));
	delete[] lengths; 
	delete[] lengths_cu;
	delete[] lengths_mu;
	delete[] nodes;
	delete[] biases;
	delete[] weights;
	delete[] result; 
}

int main() {
	start_time = 0; start_time = time_millis();
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	max_threads_per_block = prop.maxThreadsPerBlock; 

	double gpu_cpu_times[2];
	
#if defined(SMALL_WIDTH_INCREASING_LENGTH)
	std::ofstream small_width_increasing_length_gpu_file("GPU small width increasing length.csv");
	std::ofstream small_width_increasing_length_cpu_file("CPU small width increasing length.csv");
	small_width_increasing_length_gpu_file << "length,time" << std::endl;
	small_width_increasing_length_cpu_file << "length,time" << std::endl;
	for (int length = 2; length <= 50; length++) {
		test(length, 10, 10000, gpu_cpu_times);
		small_width_increasing_length_gpu_file << length << "," << gpu_cpu_times[0] << std::endl; 
		small_width_increasing_length_cpu_file << length << "," << gpu_cpu_times[1] << std::endl; 
	}
	small_width_increasing_length_gpu_file.close(); 
	small_width_increasing_length_cpu_file.close(); 
#endif 

#if defined(LARGE_WIDTH_INCREASING_LENGTH)
	std::ofstream large_width_increasing_length_gpu_file("GPU large width increasing length.csv");
	std::ofstream large_width_increasing_length_cpu_file("CPU large width increasing length.csv");
	large_width_increasing_length_gpu_file << "length,time" << std::endl;
	large_width_increasing_length_cpu_file << "length,time" << std::endl; 
	for (int length = 2; length <= 8; length++) {
		test(length, 10000, 10, gpu_cpu_times);
		large_width_increasing_length_gpu_file << length << "," << gpu_cpu_times[0] << std::endl;
		large_width_increasing_length_cpu_file << length << "," << gpu_cpu_times[1] << std::endl;
	}
	large_width_increasing_length_gpu_file.close();
	large_width_increasing_length_cpu_file.close();
#endif 

#if defined(INCREASING_WIDTH_SMALL_LENGTH)
	std::ofstream increasing_width_small_length_gpu_file("GPU increasing width small length.csv");
	std::ofstream increasing_width_small_length_cpu_file("CPU increasing width small length.csv");
	increasing_width_small_length_gpu_file << "width,time" << std::endl;
	increasing_width_small_length_cpu_file << "width,time" << std::endl;
	for (int width = 2; width <= 10000;) {
		test(5, width, 10, gpu_cpu_times);
		increasing_width_small_length_gpu_file << width << "," << gpu_cpu_times[0] << std::endl;
		increasing_width_small_length_cpu_file << width << "," << gpu_cpu_times[1] << std::endl;
		if (width < 10) width += 2;
		else if (width < 200) width += 10;
		else width += 100;
	}
	increasing_width_small_length_gpu_file.close();
	increasing_width_small_length_cpu_file.close();
#endif 

#if defined(INCREASING_WIDTH_LARGE_LENGTH) 
	std::ofstream increasing_width_large_length_gpu_file("GPU increasing width large length.csv");
	std::ofstream increasing_width_large_length_cpu_file("CPU increasing width large length.csv");
	increasing_width_large_length_gpu_file << "width,time" << std::endl;
	increasing_width_large_length_cpu_file << "width,time" << std::endl;
	for (int width = 2; width <= 10000;) {
		test(20, width, 3, gpu_cpu_times);
		increasing_width_large_length_gpu_file << width << "," << gpu_cpu_times[0] << std::endl;
		increasing_width_large_length_cpu_file << width << "," << gpu_cpu_times[1] << std::endl;
		if (width < 10) width += 8;
		else if (width < 100) width += 10; 
		else if (width < 1000) width += 100;
		else width += 1000;
	}
	increasing_width_large_length_gpu_file.close();
	increasing_width_large_length_cpu_file.close();
#endif 
}