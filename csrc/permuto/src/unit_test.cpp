/** @file   unit_test.cpp
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  Directly runnable unit test for permutohedral encoding.
*/

#include <stdint.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <chrono>

#include <torch/torch.h>

#include <permuto/permuto.h>

/*
cd nr3d_lib

mkdir -p build/unit_test_permuto

nvcc -j12 \
-I/home/guojianfei/ai_ws/neuralsim_dev/nr3d_lib/csrc/permuto/include \
-I/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/include \
-I/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/include/torch/csrc/api/include \
-I/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/include/TH \
-I/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/include/THC \
-I/usr/local/cuda/include \
-c -c csrc/permuto/src/permuto_cuda.cu \
-o build/unit_test_permuto/lib.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -std=c++14 --extended-lambda --expt-relaxed-constexpr -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 -Xcompiler=-mf16c -Xcompiler=-Wno-float-conversion -Xcompiler=-fno-strict-aliasing -Xcudafe=--diag_suppress=unrecognized_gcc_pragma -D_GLIBCXX_USE_CXX11_ABI=0

g++ -pthread -fPIC -g -Wl,--sysroot=/ \
-Wl,--no-as-needed \
-Wl,-rpath=/usr/local/cuda/lib64 \
-Wl,-rpath=/home/guojianfei/anaconda3/envs/ml/lib \
-Wl,-rpath=/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/lib \
csrc/permuto/src/unit_test.cpp build/unit_test_permuto/lib.o -o build/unit_test_permuto/fwd \
-L/usr/local/cuda/lib64 \
-L/home/guojianfei/anaconda3/envs/ml/lib \
-L/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/lib \
-lc10 -ltorch -ltorch_cpu -lcudart -lc10_cuda -ltorch_cuda_cu -ltorch_cuda_cpp \
-I/home/guojianfei/ai_ws/neuralsim_dev/nr3d_lib/csrc/permuto/include \
-I/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/include \
-I/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/include/torch/csrc/api/include \
-I/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/include/TH \
-I/home/guojianfei/anaconda3/envs/ml/lib/python3.8/site-packages/torch/include/THC \
-I/usr/local/cuda/include \
-std=c++14 \
-D_GLIBCXX_USE_CXX11_ABI=0

*/


int main(int argc, char* argv[]) {
	int nruns = 20;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		std::size_t pos;

		if ((pos = arg.find("-nruns=")) != std::string::npos) {
			std::stringstream ss(arg.substr(pos + 7));
			ss >> nruns;
		}
	}

	//---- Setup
	const uint32_t hashmap_size = 1<<18;
	const uint32_t n_levels = 24;
	const uint32_t n_input_dim = 7;
	const double level_scale = 1.32;

	auto res_list = std::vector<double>(n_levels);
	double res = 16.0; 
	for (uint32_t lvl=0; lvl < n_levels; ++lvl) {
		res_list[lvl] = res;
		std::cout<<"level:"<<lvl<<" res="<<res<<std::endl;
		res *= level_scale; 
	}
	auto n_feats_list = std::vector<int32_t>(n_levels, 2);
	auto meta = permuto::PermutoEncMeta((int32_t)n_input_dim, (int32_t)hashmap_size, res_list, n_feats_list);

	at::Tensor lattice_values = at::randn({(int64_t)meta.n_params}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
	at::Tensor level_random_shifts = 10.0 * at::randn({(int64_t)meta.n_levels, (int64_t)meta.n_dims_to_encode}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

	//---- Start testing
	const uint32_t batch_size = 3653653;
	at::Tensor positions = at::rand({(int64_t)batch_size, (int64_t)n_input_dim}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
	at::Tensor encoded = permuto::permuto_enc_fwd(meta, positions, lattice_values, level_random_shifts, at::nullopt, at::nullopt, at::nullopt, at::nullopt); 
	cudaDeviceSynchronize();
	std::vector<int64_t> sizes(encoded.sizes().begin(), encoded.sizes().end()); 
	std::cout << "The encoded should have size: [";
	for (uint32_t di=0; di<sizes.size(); ++di) std::cout << sizes[di] << (di==sizes.size()-1?"":","); 
	std::cout << "]" <<std::endl;

	//---- Simple benchmarking
	std::cout << "Start " << nruns << " runs..." <<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	for (int run=0; run<nruns; ++run) {
		permuto::permuto_enc_fwd(meta, positions, lattice_values, level_random_shifts, at::nullopt, at::nullopt, at::nullopt, at::nullopt); 
	}
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end - start;
	double ms_iter = elapsed.count() / (double)nruns;
	std::cout <<"Done " << nruns << " runs, each took " << ms_iter <<" ms average." <<std::endl; 

	return 1;
}