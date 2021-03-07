#include <iostream>
#include <math.h>
#include <ctime>

#define CONVERGENCE 5
#define NONCONVERGENCE 5

void PrintVector(double* vector, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << vector[i] << " ";
    }
}

void MatrixAndVectorMultiplication(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = 0;
        for (int j = 0; j < N; j++) {
            C[i] += A[i * N + j] * B[j];
        }
    }
}

double ScalarVectorsMultiplication(double* A, double* B, int N) {
    double C = 0;
    for (int i = 0; i < N; i++) {
        C += A[i] * B[i];
    }
    return C;
}

void ScalarAndVectorMultiplication(double A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A * B[i];
    }
}

void VectorsSubstruction(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] - B[i];
    }
}

int CheckOnRepitition(double result, double e, int convergentMatrixRepetition) {
    if (result < e) {
        convergentMatrixRepetition++;
    }
    else {
        convergentMatrixRepetition = 0;
    }
    return convergentMatrixRepetition;
}

void FillVectorWithZero(double* vector, int N) {
    for (int i = 0; i < N; i++) {
        vector[i] = 0;
    }
}

double SquaresSum(double* v, int N) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

double EndCycleCriteria(double* A, double* x, double* b, double bVectorLenght, int N) {
    double vector[N];
    MatrixAndVectorMultiplication(A, x, vector, N);
    VectorsSubstruction(vector, b, vector, N);
    double result = SquaresSum(vector, N) / bVectorLenght;
    return result;
}

void PrintMatrixA(double* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << " A[] " << std::endl << std::endl;;
}

void FillMatrixA(double* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[i * N + j] = 2;
            }
            else {
                A[i * N + j] = 1;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    auto* A = new double[N * N];
    auto* x = new double[N];
    auto* b = new double[N];
    auto* y = new double[N];
    auto* Ay = new double[N];
    auto* ty = new double[N];
    auto* u = new double[N];
    double t;
    double firstScalar;
    double secondScalar;
    double e = 1e-008;
    int convergentMatrixRepetition = 0;
    int notConvergentMatrixCounter = 0;
    double previousResult = 0;
    double bVectorLenght = 0;
    bool nonConvergation;

    FillMatrixA(A, N);
    //PrintMatrixA(A, N);

    for (int i = 0; i < N; i++) {
        u[i] = sin((2 * 3.14159 * i) / N);
        std::cout << u[i] << " ";
    }
    std::cout << " u[]" << std::endl << std::endl;

    MatrixAndVectorMultiplication(A, u, b, N);

    //PrintVector(b, N);
    //std::cout << " b[] " << std::endl << std::endl;

    bVectorLenght = SquaresSum(b, N);

    FillVectorWithZero(x, N); //x_0 = {0};

    double result = EndCycleCriteria(A, x, b, bVectorLenght, N);

    int amountOfIterations = 0;
    long double startMeasureTime = clock();
    while ((result > e) && (convergentMatrixRepetition < CONVERGENCE)) {
        amountOfIterations++;
        convergentMatrixRepetition = CheckOnRepitition(result, e, convergentMatrixRepetition);
        previousResult = result;

        MatrixAndVectorMultiplication(A, x, y, N); //y_n = Ax_n
        VectorsSubstruction(y, b, y, N); //y_n = Ax_n - b
        MatrixAndVectorMultiplication(A, y, Ay, N); //Ay_n
        firstScalar = ScalarVectorsMultiplication(y, Ay, N); //(y_n, Ay_n)
        secondScalar = ScalarVectorsMultiplication(Ay, Ay, N);
        t = firstScalar / secondScalar;
        ScalarAndVectorMultiplication(t, y, ty, N);
        VectorsSubstruction(x, ty, x, N); // x_n+1 = x_n - t_ny_n

        result = EndCycleCriteria(A, x, b, bVectorLenght, N);
        std::cout << "Result is: " << result << std::endl;
        if (previousResult < result) {
            notConvergentMatrixCounter++;
            if (notConvergentMatrixCounter > NONCONVERGENCE || previousResult == INFINITY) {
                nonConvergation = true;
                break;
            }
        }
        else {
            notConvergentMatrixCounter = 0;
        }
    }
    long double endMeasureTime = clock();

    if (nonConvergation) {
        std::cout << "Impossible task! Matrix is not convergent!" << std::endl;
        delete[](u);
        delete[](ty);
        delete[](Ay);
        delete[](y);
        delete[](b);
        delete[](x);
        delete[](A);
        return 0;
    }
    PrintVector(x, N);
    std::cout << " x[] " << std::endl << std::endl;

    std::cout << "Total time is: " << (endMeasureTime - startMeasureTime) / 1000000 << " seconds" << std::endl;
    std::cout << "Iterations: " << amountOfIterations + 1 << std::endl;
    delete[](u);
    delete[](ty);
    delete[](Ay);
    delete[](y);
    delete[](b);
    delete[](x);
    delete[](A);
    return 0;
}


