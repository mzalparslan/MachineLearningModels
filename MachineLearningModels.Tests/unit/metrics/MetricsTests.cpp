#include "pch.h"
#include "metrics.h"

TEST(MetricsTest, MeanSquaredErrorAndRootMeanSquaredError) {
    const std::vector<double> predictions{ 3.0, 5.0, 7.0 };
    const std::vector<double> targets{ 2.0, 5.0, 9.0 };

    // Errors: 1, 0, -2 -> squared: 1, 0, 4 -> mean = 5/3.
    const double mse = metrics::meanSquaredError(predictions, targets);
    EXPECT_NEAR(mse, 5.0 / 3.0, 1e-9);

    EXPECT_NEAR(
        metrics::rootMeanSquaredError(predictions, targets),
        std::sqrt(mse),
        1e-9);
}

TEST(MetricsTest, MeanAbsoluteError) {
    const std::vector<double> predictions{ 3.0, 5.0, 7.0 };
    const std::vector<double> targets{ 2.0, 5.0, 9.0 };

    // Absolute errors: 1, 0, 2 -> mean = 1.0.
    EXPECT_NEAR(metrics::meanAbsoluteError(predictions, targets), 1.0, 1e-9);
}

TEST(MetricsTest, RSquaredIsOneForPerfectPredictions) {
    const std::vector<double> targets{ 1.0, 2.0, 3.0, 4.0 };
    EXPECT_NEAR(metrics::rSquared(targets, targets), 1.0, 1e-9);
}

TEST(MetricsTest, RSquaredIsZeroWhenTargetsAreConstant) {
    const std::vector<double> predictions{ 1.0, 2.0, 3.0 };
    const std::vector<double> targets{ 5.0, 5.0, 5.0 };

    // Total sum of squares is zero, which is conventionally undefined;
    // this implementation returns 0.0 rather than dividing by zero.
    EXPECT_DOUBLE_EQ(metrics::rSquared(predictions, targets), 0.0);
}

TEST(MetricsTest, WithinToleranceRatioCountsHitsAndMisses) {
    const std::vector<double> predictions{ 10.5, 20.0, 5.0 };
    const std::vector<double> targets{ 10.0, 20.0, 100.0 };

    // Sample 0: |10.5-10| = 0.5 <= 0.1*10 = 1.0 -> within tolerance.
    // Sample 1: exact match -> within tolerance.
    // Sample 2: |5-100| = 95 > 0.1*100 = 10 -> outside tolerance.
    EXPECT_NEAR(
        metrics::withinToleranceRatio(predictions, targets, 0.1),
        2.0 / 3.0,
        1e-9);
}

TEST(MetricsTest, MismatchedSizesThrow) {
    const std::vector<double> predictions{ 1.0, 2.0 };
    const std::vector<double> targets{ 1.0 };

    EXPECT_THROW(metrics::meanSquaredError(predictions, targets), std::invalid_argument);
    EXPECT_THROW(metrics::meanAbsoluteError(predictions, targets), std::invalid_argument);
    EXPECT_THROW(metrics::rSquared(predictions, targets), std::invalid_argument);
    EXPECT_THROW(metrics::withinToleranceRatio(predictions, targets), std::invalid_argument);
}

TEST(MetricsTest, BinaryCrossEntropyFromLogitMatchesProbabilityForm) {
    // For target = 1, loss should be -log(sigmoid(logit)).
    const double logit = 2.0;
    const double probability = 1.0 / (1.0 + std::exp(-logit));

    EXPECT_NEAR(
        metrics::binaryCrossEntropyFromLogit(logit, 1.0),
        -std::log(probability),
        1e-9);
}

TEST(MetricsTest, BinaryCrossEntropyPenalizesConfidentWrongPredictions) {
    const std::vector<double> confidentCorrect{ 0.99 };
    const std::vector<double> confidentWrong{ 0.01 };
    const std::vector<double> targets{ 1.0 };

    const double lowLoss = metrics::binaryCrossEntropy(confidentCorrect, targets);
    const double highLoss = metrics::binaryCrossEntropy(confidentWrong, targets);

    EXPECT_LT(lowLoss, highLoss);
}

TEST(MetricsTest, ClassificationRatesComputeAccuracyPrecisionRecallF1) {
    // 2 true positives, 1 true negative, 1 false positive, 1 false negative.
    const std::vector<bool> predicted{ true, true, false, true, false };
    const std::vector<bool> expected{ true, true, false, false, true };

    const BinaryClassificationMetrics result =
        metrics::classificationRates(predicted, expected);

    EXPECT_NEAR(result.accuracy, 3.0 / 5.0, 1e-9);
    EXPECT_NEAR(result.precision, 2.0 / 3.0, 1e-9);
    EXPECT_NEAR(result.recall, 2.0 / 3.0, 1e-9);
    EXPECT_NEAR(result.f1Score, 2.0 / 3.0, 1e-9);
}

TEST(MetricsTest, ClassificationRatesReturnZeroForDegenerateDenominators) {
    const std::vector<bool> allNegativePredictions{ false, false };
    const std::vector<bool> allNegativeExpected{ false, false };

    const BinaryClassificationMetrics result =
        metrics::classificationRates(allNegativePredictions, allNegativeExpected);

    EXPECT_DOUBLE_EQ(result.precision, 0.0);
    EXPECT_DOUBLE_EQ(result.recall, 0.0);
    EXPECT_DOUBLE_EQ(result.f1Score, 0.0);
    EXPECT_DOUBLE_EQ(result.accuracy, 1.0);
}
