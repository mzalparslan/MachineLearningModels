# MachineLearningModels

MachineLearningModels is an educational C++ project that implements core
machine-learning algorithms without relying on a machine-learning framework.
It demonstrates feature preprocessing, gradient-descent optimization, linear
regression, logistic binary classification, neural networks, data loading,
evaluation output, and automated testing.

Repository contains four projects:

- **MachineLearningModels** — reusable static library.
- **MachineLearningModels.Examples** — example regression, classification, and
  neural-network experiments (in `src/experiments/`).
- **MachineLearningModels.Tests** — Google Test unit-test project.
- **MachineLearningModels.Benchmarks** — performance comparisons of parallel and/or sequential execution strategies.


## Features

### Models and Pipelines

- Multiple linear regression
- Logistic binary classification
- BasicNeuralNetwork — a two-layer (one hidden layer) sigmoid network
- DeepNeuralNetwork — a feedforward sigmoid network with an arbitrary
  number of layers, trained by per-sample backpropagation
- Regression pipeline combining a scaler, optimizer, and model
- Binary-classification pipeline combining a scaler, optimizer, and model

### Optimization

- Batch gradient descent
- Mini-batch gradient descent
- Stochastic gradient descent
- Configurable learning rate, epoch count, L2 regularization, and bias regularization

### Feature Scaling

- Min-max scaling
- Mean normalization
- Z-score standardization
- Maximum absolute scaling
- Robust scaling using the median and interquartile range
- No-scaling policy

Scaling parameters are fitted only on training data. The fitted parameters are
then reused to transform prediction and test data, preventing data leakage.

### Execution Strategies

Linear-output calculations support configurable standard-library execution modes:

- `std::inner_product`
- Sequenced `std::transform_reduce`
- Vectorized execution
- Parallel execution
- Parallel-vectorized execution

Execution strategy can be selected when constructing a model. Sequential
execution is generally preferable for ordinary feature vectors, while parallel
execution may benefit unusually large vectors. Actual performance depends on
the processor, compiler, and standard-library implementation.

### Supporting Components

- Activation functions and derivatives (Sigmoid, TanH, ReLU, Leaky ReLU,
  PReLU, ELU) used by the neural network models
- Numeric CSV dataset loader
- Reproducible training/test splitting, including stratified splits and
  stratified k-fold cross-validation folds
- Regression and classification CSV output writers
- Stream-style logger with configurable severity levels
- Benchmark and scope-based benchmark timers
- Unit tests for the main components

## Example datasets

The examples run on these datasets, in `resources/`:

- `classification.csv` and `lineardata.csv` — synthetic datasets for the
  pipelines.
- `breast_cancer.csv` — Breast Cancer Wisconsin (Diagnostic) from UCI,
  569 samples and 30 features. Used for a stratified 5-fold comparison of
  logistic regression and both neural networks.
- `external/creditcard.csv` — optional, **not stored in git** (150 MB).
  The ULB credit card fraud dataset: 284,807 transactions, 492 frauds. If
  the file is missing, that experiment logs a warning and is skipped.

To download the fraud dataset (a public copy of the Kaggle data), run from
the repository root:

```bash
mkdir -p resources/external
curl -L -o resources/external/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
sed -i 's/"//g' resources/external/creditcard.csv
```

The `sed` step removes the quotes around the `Class` column, which
`CsvDataLoader` does not support. On Windows, run these in Git Bash or WSL.

## Example results

Both experiments use stratified 5-fold cross-validation (seed 42), fit a
Z-score scaler on each fold's training part only, and classify at a 0.5
threshold. Values are mean ± standard deviation across the five folds,
measured on one machine in a Release build; timings will vary.

### Breast Cancer Wisconsin (569 samples, 30 features)

| Model | Accuracy | Precision | Recall | F1 | Cross-entropy |
|---|---|---|---|---|---|
| `LogisticBinaryClassifier` (batch GD, 1000 epochs) | 0.968 ± 0.021 | 0.975 ± 0.018 | 0.938 ± 0.049 | 0.956 ± 0.030 | 0.078 ± 0.039 |
| `BasicNeuralNetwork` (8 hidden units, 200 epochs) | 0.974 ± 0.021 | 0.967 ± 0.027 | 0.962 ± 0.036 | 0.964 ± 0.028 | 0.112 ± 0.092 |
| `DeepNeuralNetwork` (30-16-8-1, 200 epochs) | 0.975 ± 0.018 | 0.968 ± 0.034 | 0.967 ± 0.032 | 0.967 ± 0.024 | 0.123 ± 0.084 |

The three models are effectively tied: accuracy and recall differ by less
than the fold-to-fold spread. Logistic regression has the lowest
cross-entropy, so the networks make similar decisions with more
overconfident probabilities.

### Credit card fraud (284,807 transactions, 492 frauds = 0.17%)

An unweighted baseline: no class weighting or undersampling, `Time` column
dropped (29 features), and a fixed 0.5 threshold. Accuracy is not reported,
because always predicting "genuine" already scores 99.83%. All three models
use per-sample updates with learning rate 0.05 and 20 epochs.

| Model | Precision | Recall | F1 | Cross-entropy | Time (5 folds) |
|---|---|---|---|---|---|
| `LogisticBinaryClassifier` (SGD) | 0.865 ± 0.076 | 0.637 ± 0.281 | 0.684 ± 0.245 | 0.0098 ± 0.0015 | 6.6 s |
| `BasicNeuralNetwork` (8 hidden units) | 0.879 ± 0.039 | 0.750 ± 0.033 | 0.808 ± 0.017 | 0.0031 ± 0.0007 | 8.8 s |
| `DeepNeuralNetwork` (29-16-8-1) | 0.867 ± 0.081 | 0.733 ± 0.125 | 0.783 ± 0.056 | 0.0032 ± 0.0006 | 23 s |

The networks reach about three times lower cross-entropy than logistic
regression, a gap far larger than the fold-to-fold spread. Their higher F1
is suggestive but not conclusive, because the logistic baseline is unstable
across folds (recall ± 0.28). The two networks are tied within noise, and
the deeper one is about 2.6 times slower.

## Performance Benchmarks

`MachineLearningModels.Benchmarks` compares the available linear-output
execution strategies across different feature-vector sizes.

Build and run it with Visual Studio using:

```text
Configuration: Release
Platform: x64
Debug → Start Without Debugging
```

## Requirements

- A C++20-compatible compiler
- GNU Make for the Linux build
- Google Test for building the test project
- Intel TBB (`libtbb-dev` on Ubuntu/WSL) -- required to link, not optional
- Visual Studio 2022 with C++ desktop-development tools for the Windows solution

On Ubuntu or WSL, install the required development packages with:

```bash
sudo apt update
sudo apt install build-essential libgtest-dev libtbb-dev
```

`libtbb-dev` is required because libstdc++ routes the parallel execution
policies (`std::execution::par`, `par_unseq`) through Intel TBB, and the
Makefile links every target against it. Skip this package and the build
fails at the link step with `cannot find -ltbb`, not a silent runtime
fallback to sequential execution -- so if you hit that error, this is the
fix.

## Building on Linux or WSL

From the repository root, build all three projects:

```bash
make -j"$(nproc)"
```

Individual targets are also available:

```bash
make library
make examples
make tests
make test
make benchmarks
```

Build an optimized configuration with:

```bash
make BUILD=release -j"$(nproc)"
```

Remove generated Makefile outputs with:

```bash
make clean
```

Debug outputs are written to:

```text
build/debug/lib/libMachineLearningModels.a
build/debug/bin/MachineLearningModels.Examples
build/debug/bin/MachineLearningModels.Tests
```

The Makefile copies the example CSV datasets from `resources/` into the binary
directory. Run the examples from that directory so their relative resource
paths resolve correctly:

```bash
cd build/debug/bin
./MachineLearningModels.Examples
```

The program writes predictions to `binaryOutput.csv` and `linearOutput.csv`.

## Building with Visual Studio

1. Open `MachineLearningModels.sln` in Visual Studio 2022.
2. Restore NuGet packages for `MachineLearningModels.Tests` (its
   `packages.config` is tracked, but the packages it restores into
   `packages/` are not). Visual Studio does this automatically on build;
   if it doesn't, right-click the solution and choose **Restore NuGet
   Packages**.
3. Select the desired configuration and platform, such as `Debug | x64`.
4. Build the solution.
5. Set **MachineLearningModels.Examples** as the startup project.
6. Set its debugging working directory to `$(TargetDir)`.

The example datasets must be copied from `resources/` to `$(TargetDir)` before
the example program is launched. The project can perform this through its build
configuration.

## Running tests

On Linux or WSL:

```bash
make test
```

In Visual Studio, open **Test Explorer** and select **Run All Tests**.

The tests cover preprocessing strategies, optimizers, models, pipelines, CSV
utilities, output writers, logging, and benchmark timers.

## Project structure

```text
MachineLearningModels/
|-- MachineLearningModels/            # Static library
|   |-- include/
|   |   |-- data/
|   |   |-- metrics/
|   |   |-- models/
|   |   |-- optimization/
|   |   |-- pipelines/
|   |   |-- preprocessing/
|   |   `-- utilities/
|   |-- src/
|   |   |-- metrics/
|   |   `-- utilities/
|-- MachineLearningModels.Examples/   # Example application
|-- MachineLearningModels.Tests/      # Google Test suites
|   |-- unit/
|   |   |-- data/
|   |   |-- metrics/
|   |   |-- models/
|   |   |-- optimization/
|   |   |-- pipelines/
|   |   |-- preprocessing/
|   |   `-- utilities/
|-- MachineLearningModels.Benchmarks/ # Performance benchmarks
|-- resources/                        # Example CSV datasets
|-- Makefile                          # Linux/WSL build
`-- MachineLearningModels.sln         # Visual Studio solution
```

## Example design

Algorithms are selected at compile time through template parameters. For
example, a binary-classification pipeline can combine Z-score scaling with
batch gradient descent:

```cpp
using Pipeline = BinaryClassificationPipeline<
    float,
    ZScoreScaler<float>,
    BatchGradientDescent<float>>;
```

This makes the model, optimizer, and preprocessing policies independently
replaceable without runtime polymorphism in the pipeline.

## Scope

This project is intended for learning, experimentation, and demonstration of
modern C++ design applied to fundamental machine-learning algorithms. It is not
intended to replace established production machine-learning libraries.
