#include <iostream>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <iomanip>

template<typename T> T inner_product(const std::vector<T>& x, const std::vector<T>& y){
	int dimention = std::min(x.size(), y.size());
	T res = 0;
	for(int i = 0; i < dimention; ++i){
		res += x[i]*y[i];
	}
	return res;
}

template<typename T> void learn(std::vector<T>& weights, const std::vector<std::vector<T>>& input, const std::vector<int>& output, const T lambda, const T eta){
	std::vector<T> grad(weights.size());
	for(int i = 0; i < input.size(); ++i){
		T tmp = std::exp(-output[i] * inner_product(weights, input[i]));
		tmp = tmp / (1 + tmp) * (-output[i]);
		for(int j=0; j < input[i].size(); ++j){
			grad[j] += tmp * input[i][j];
		}
	}
	for(int i = 0; i < weights.size(); ++i){
		grad[i] += 2 * lambda * weights[i];
	}
	for(int i = 0; i < weights.size(); ++i){
		weights[i] -= eta * grad[i];
	}
	return;
}

template<typename T> T score(const std::vector<T>& weights, const std::vector<std::vector<T>>& input, const std::vector<int>& output, const T lambda){
	double res = 0.0;
	for(int i = 0; i < input.size(); ++i){
		res += std::log(1 + std::exp(-output[i] * inner_product(weights, input[i])));
	}
	res += lambda * inner_product(weights, weights);
	return res;
}

int main(int argc, char const *argv[]) {
	std::cout << std::setprecision(10)<< std::fixed;

	// dimention of a input vector and a parameter
	int dimention = 2;
	if(argc >= 2){
		dimention = std::atoi(argv[1]);
	}

	// iterator
	int iter = 10;
	if(argc >= 3){
		iter = std::atoi(argv[2]);
	}

	// learning rate
	double eta = 0.1;
	if(argc >= 4){
		eta = std::atof(argv[3]);
	}

	// a parameter of regularization
	double lambda = 0.01;
	if(argc >= 5){
		lambda = std::atof(argv[4]);
	}

	std::vector<std::vector<double>> input(1);
	std::vector<int> output(1);

	// a parameter of the model
	std::vector<double> weights(dimention);

	// a parameter of the optimal model
	std::vector<double> optimal_weights(dimention);
	double optimal_value = score(optimal_weights, input, output, lambda);

	while(--iter >= 0){
		learn(weights, input, output, lambda, eta);
		double error = score(weights, input, output, lambda) - optimal_value;
		std::cout << error << '\n';
	}

	return 0;
}
