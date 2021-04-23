#include <iostream>
#include <math.h>
#include <ctime>
#include <omp.h>

#define CONVERGENCE 5
#define NONCONVERGENCE 5

void PrintVec(double* vector, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << vector[i] << " ";
    }
}

void MatVecMul(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = 0;
        for (int j = 0; j < N; j++) {
            C[i] += A[i * N + j] * B[j];
        }
    }
}

double ScalProduct(double* A, double* B, int N) {
    double C = 0;
    for (int i = 0; i < N; i++) {
        C += A[i] * B[i];
    }
    return C;
}

void VecByNumMul(double A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A * B[i];
    }
}

void VecSub(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] - B[i];
    }
}

void FillU(double *u, int N) { //YEY
    for (int i = 0; i < N; i++) {
        u[i] = sin((2 * 3.14159 * i) / N);
    }
}

void FillMat(double* A, int N) {
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
    auto* currX = new double[N]; // xn+1
    auto* prevX = new double[N]; // xn
    auto* b = new double[N];
    auto* y = new double[N];
    auto* Atmp = new double[N];
    auto* tauY = new double[N];
    auto* u = new double[N];
    double tau;
    double firstScalar;
    double secondScalar;
    double e = 1e-008;
    double yVecLenght;
    double bVecLenght;

    double result = 1;
    double prevResult = 1;
    bool diverge = false;
    int divergenceCount = 0; //повторения расхождения матрицы
    int convergentMatRepetition = 0; //повторения сходимости матрицы
    int cycleIterations = 0;

    FillMat(A, N);
    FillU(u, N);

    MatVecMul(A, u, b, N);

    std::fill(currX, currX + N, 0); //x_n+1 = {0};
    std::fill(prevX, prevX + N, 0); //x_n = {0};

    bVecLenght = sqrt(ScalProduct(b, b, N));

    long double startMeasureTime = clock();
    while ((result > e) && (convergentMatRepetition < CONVERGENCE)) {
        if (result < e) {
            convergentMatRepetition++;
        }
        else {
            convergentMatRepetition = 0;
        }

        MatVecMul(A, prevX, y, N); //y_n = Ax_n
        yVecLenght = sqrt(ScalProduct(y, y, N));
        VecSub(y, b, y, N); //y_n = Ax_n - b
        MatVecMul(A, y, Atmp, N); //Ay_n
        firstScalar = ScalProduct(y, Atmp, N); //(y_n, Ay_n)
        secondScalar = ScalProduct(Atmp, Atmp, N);
        tau = firstScalar / secondScalar;
        VecByNumMul(tau, y, tauY, N);
        VecSub(prevX, tauY, currX, N); // x_n+1 = x_n - t_ny_n
        result = yVecLenght / bVecLenght;

        if (prevResult < result) {
            divergenceCount++;
            if (divergenceCount > NONCONVERGENCE || prevResult == INFINITY) {
                diverge = true;
                break;
            }
        }
        else {
            divergenceCount = 0;
        }
        prevResult = result;
        for (int i = 0; i < N; i++) {
            prevX[i] = currX[i];
        }
        cycleIterations++;
    }
    long double endMeasureTime = clock();

    if (diverge) {
        std::cout << "Impossible task! Matrix is not convergent!" << std::endl;
        delete[](u);
        delete[](tauY);
        delete[](Atmp);
        delete[](y);
        delete[](b);
        delete[](prevX);
        delete[](currX);
        delete[](A);
        return 0;
    }
    PrintVec(u, N);
    std::cout << " u[] " << std::endl << std::endl;
    PrintVec(currX, N);
    std::cout << " x[] " << std::endl << std::endl;

    std::cout << "Total time is: " << (endMeasureTime - startMeasureTime) / 1000000 << " seconds" << std::endl;
    std::cout << "Iterations: " << cycleIterations + 1 << std::endl;
    delete[](u);
    delete[](tauY);
    delete[](Atmp);
    delete[](y);
    delete[](b);
    delete[](prevX);
    delete[](currX);
    delete[](A);
    return 0;
}

