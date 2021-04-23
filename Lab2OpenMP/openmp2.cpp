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
    int threadCount = atoi(argv[2]);
    int chunkSize = N / threadCount * atof(argv[3]);
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

    omp_set_num_threads(threadCount);

    FillMat(A, N);
    FillU(u, N);

    MatVecMul(A, u, b, N);

    std::fill(currX, currX + N, 0); //x_n+1 = {0};
    std::fill(prevX, prevX + N, 0); //x_n = {0};

    bVecLenght = sqrt(ScalProduct(b, b, N));

    long double startMeasureTime = clock();
#pragma omp parallel firstprivate(result, prevResult, diverge, divergenceCount, convergentMatRepetition)
    {
        while ((result > e) && (convergentMatRepetition < CONVERGENCE)) {
            if (result < e) {
                convergentMatRepetition++;
            } else {
                convergentMatRepetition = 0;
            }

    #pragma omp for schedule(guided, chunkSize)
            for (int i = 0; i < N; i++) { //y_n = Ax_n
                y[i] = 0;
                for (int j = 0; j < N; j++) {
                    y[i] += A[i * N + j] * prevX[j];
                }
            }

    #pragma omp single
            {
                yVecLenght = 0;
            }

    #pragma omp for schedule(guided, chunkSize) reduction(+:yVecLenght) //yVecLenght = sqrt(ScalProduct(y, y, N));
            for (int i = 0; i < N; i++) {
                yVecLenght += y[i] * y[i];
            }

    #pragma omp single
            {
                yVecLenght = sqrt(yVecLenght);
            }

    #pragma omp for schedule(guided, chunkSize) //y_n = Ax_n - b
            for (int i = 0; i < N; i++) {
                y[i] = y[i] - b[i];
            }

    #pragma omp for schedule(guided, chunkSize) //Ay_n
            for (int i = 0; i < N; i++) {
                Atmp[i] = 0;
                for (int j = 0; j < N; j++) {
                    Atmp[i] += A[i * N + j] * y[j];
                }
            }

    #pragma omp single
            {
                firstScalar = 0;
                secondScalar = 0;
            }

    #pragma omp for schedule(guided, chunkSize) reduction(+:firstScalar)//(y_n, Ay_n)
            for (int i = 0; i < N; i++) {
                firstScalar += y[i] * Atmp[i];
            }

    #pragma omp for schedule(guided, chunkSize) reduction(+:secondScalar)//(Ay_n, Ay_n)
            for (int i = 0; i < N; i++) {
                secondScalar += Atmp[i] * Atmp[i];
            }

    #pragma omp single
            {
                tau = firstScalar / secondScalar;
            }

    #pragma omp for schedule(guided, chunkSize) //VecByNumMul(tau, y, tauY, N);
            for (int i = 0; i < N; i++) {
                tauY[i] = tau * y[i];
            }

    #pragma omp for schedule(guided, chunkSize) // x_n+1 = x_n - t_ny_n
            for (int i = 0; i < N; i++) {
                currX[i] = prevX[i] - tauY[i];
            }

            result = yVecLenght / bVecLenght;

            if (prevResult < result) {
                divergenceCount++;
                if (divergenceCount > NONCONVERGENCE || prevResult == INFINITY) {
                    diverge = true;
                    break;
                }
            } else {
                divergenceCount = 0;
            }

    #pragma omp single
            {
                prevResult = result;
            }

    #pragma omp for schedule(guided, chunkSize)
            for (int i = 0; i < N; i++) {
                prevX[i] = currX[i];
            }


    #pragma omp single
            {
                cycleIterations++;
            }
        }
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


