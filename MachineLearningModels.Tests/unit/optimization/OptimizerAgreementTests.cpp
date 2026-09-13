#include "pch.h"
#include "BatchGradientDescent.h"
#include "MiniBatchGradientDescent.h"
#include "StochasticGradientDescent.h"

#include <numeric>

namespace {

	// Shared linear hypothesis: prediction = dot(weights, features) + bias.
	auto LinearHypothesis() {
		return [](const std::vector<double>& features, const ModelParameters<double>& p) {
			return std::inner_product(p.weights.begin(), p.weights.end(), features.begin(), p.bias);
		};
	}

	// y = 3x + 2 over 200 samples, matching the dataset used to disprove
	// the (retracted) claim that lambda meant different things across
	// the three optimizers. x stays small (0 to ~2) so that batch
	// gradient descent's fixed learning rate below doesn't overshoot.
	std::vector<DataPoint<double>> LinearDataset() {
		std::vector<DataPoint<double>> data;
		data.reserve(200);
		for (int i = 0; i < 200; ++i) {
			const double x = static_cast<double>(i) * 0.01;
			data.push_back(DataPoint<double>{ { x }, 3.0 * x + 2.0 });
		}
		return data;
	}

} // namespace

TEST(OptimizerAgreementTest, ConvergeToSimilarParametersWithoutRegularization) {
	const std::vector<DataPoint<double>> data = LinearDataset();

	ModelParameters<double> batchParams{ { 0.0 }, 0.0 };
	GradientDescentOptions<double> batchOptions;
	batchOptions.learningRate = 0.05;
	batchOptions.epochs = 3000;
	BatchGradientDescent<double> batch;
	batch.optimize(data, batchOptions, batchParams, LinearHypothesis());

	ModelParameters<double> sgdParams{ { 0.0 }, 0.0 };
	StochasticGradientDescentOptions<double> sgdOptions;
	sgdOptions.gradientOptions.learningRate = 0.01;
	sgdOptions.gradientOptions.epochs = 300;
	StochasticGradientDescent<double> sgd;
	sgd.optimize(data, sgdOptions, sgdParams, LinearHypothesis());

	ModelParameters<double> miniParams{ { 0.0 }, 0.0 };
	MiniBatchGradientDescentOptions<double> miniOptions;
	miniOptions.gradientOptions.learningRate = 0.02;
	miniOptions.gradientOptions.epochs = 800;
	miniOptions.batchSize = 20;
	MiniBatchGradientDescent<double> mini;
	mini.optimize(data, miniOptions, miniParams, LinearHypothesis());

	EXPECT_NEAR(batchParams.weights[0], 3.0, 0.05);
	EXPECT_NEAR(sgdParams.weights[0], 3.0, 0.1);
	EXPECT_NEAR(miniParams.weights[0], 3.0, 0.1);

	EXPECT_NEAR(batchParams.bias, 2.0, 0.05);
	EXPECT_NEAR(sgdParams.bias, 2.0, 0.1);
	EXPECT_NEAR(miniParams.bias, 2.0, 0.1);
}

TEST(OptimizerAgreementTest, RegularizationReachesSameFixedPointAcrossOptimizers) {
	// All three optimizers normalize their L2 penalty by (lambda / sampleCount),
	// so despite very different per-update step scales (batch moves by 1/m,
	// mini-batch by 1/B, SGD by 1), they converge to the same regularized
	// weight -- well below the unregularized fixed point of 3.0.
	const std::vector<DataPoint<double>> data = LinearDataset();
	constexpr double lambda = 400.0;

	ModelParameters<double> batchParams{ { 0.0 }, 0.0 };
	GradientDescentOptions<double> batchOptions;
	batchOptions.learningRate = 0.05;
	batchOptions.epochs = 3000;
	batchOptions.lambda = lambda;
	BatchGradientDescent<double> batch;
	batch.optimize(data, batchOptions, batchParams, LinearHypothesis());

	ModelParameters<double> sgdParams{ { 0.0 }, 0.0 };
	StochasticGradientDescentOptions<double> sgdOptions;
	sgdOptions.gradientOptions.learningRate = 0.01;
	sgdOptions.gradientOptions.epochs = 300;
	sgdOptions.gradientOptions.lambda = lambda;
	StochasticGradientDescent<double> sgd;
	sgd.optimize(data, sgdOptions, sgdParams, LinearHypothesis());

	ModelParameters<double> miniParams{ { 0.0 }, 0.0 };
	MiniBatchGradientDescentOptions<double> miniOptions;
	miniOptions.gradientOptions.learningRate = 0.02;
	miniOptions.gradientOptions.epochs = 800;
	miniOptions.gradientOptions.lambda = lambda;
	miniOptions.batchSize = 20;
	MiniBatchGradientDescent<double> mini;
	mini.optimize(data, miniOptions, miniParams, LinearHypothesis());

	EXPECT_LT(batchParams.weights[0], 2.5);
	EXPECT_NEAR(batchParams.weights[0], sgdParams.weights[0], 0.1);
	EXPECT_NEAR(batchParams.weights[0], miniParams.weights[0], 0.1);
}
