#include <iostream>
#include <math.h>
#include "mpi.h"

#define CONVERGENCE 5
#define NONCONVERGENCE 5

void PrintVec(double* vector, int N) { //YEY
    for (int i = 0; i < N; i++) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

void MatVecMul(double *A, double *B, double *C, int *linePerProc, int N, int rank) {
    auto *tmp = new double[N];
    std::fill(tmp, tmp + N, 0);
    for (int i = 0; i < linePerProc[rank]; i++) {
        for (int j = 0; j < N; j++) {
            tmp[j] += A[i * N + j] * B[i];
        }
    }
    MPI_Allreduce(tmp, C, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](tmp);
}

double ScalProduct(double *vec1, double *vec2, int *linePerProc, int rank) {
    double sum = 0;
    for (int i = 0; i < linePerProc[rank]; ++i) {
        sum += vec1[i] * vec2[i];
    }
    double fullSum;
    MPI_Allreduce(&sum, &fullSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return fullSum;
}

void VecByNumMul(double A, double *B, double *C, int *linePerProc, int rank) { //YEY
    for (int i = 0; i < linePerProc[rank]; i++) {
        C[i] = A * B[i];
    }
}

void VecSub(double *A, double *B, double *C, int *linePerProc, int rank) {
    for (int i = 0; i < linePerProc[rank]; i++) {
        C[i] = A[i] - B[i];
    }
}

void ColumnsDistribution(int* linePerProc, int* startPoints, int size, int N) {
    int quotient = N / size; //Количество столбцов каждому процессу
    int remainder = N % size; //Если столбцы не кратны кол-ву процессов
    startPoints[0] = 0;
    std::fill(linePerProc, linePerProc + size, quotient);
    linePerProc[size - 1] += remainder;
    for (int i = 1; i < size; ++i) {
        startPoints[i] = startPoints[i - 1] + linePerProc[i - 1];
    }
}

void FillUVec(double *u, int N, int* startPoints, int* linePerProc, int rank) { //YEY
    for (int i = 0; i < linePerProc[rank]; i++) {
        u[i] = sin((2 * 3.14159 * (i + startPoints[rank])) / N);
    }
}

void FillMat(double *A, int N, int* startPoints, int* linePerProc, int rank) { //YEY
    for (int i = 0; i < linePerProc[rank]; i++) {
        for (int j = 0; j < N; j++) {
            if (i + startPoints[rank] == j) {
                A[i * N + j] = 2;
            } else {
                A[i * N + j] = 1;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int linePerProc[size];
    int startPoints[size];
    ColumnsDistribution(linePerProc, startPoints, size, N);

    auto *A = new double[linePerProc[rank] * N];
    auto *u = new double[linePerProc[rank]];
    auto* fullU = new double[N];
    auto *b = new double[linePerProc[rank]];
    auto *currX = new double[linePerProc[rank]]; // xn+1
    auto *prevX = new double[linePerProc[rank]]; // xn
    auto *fullX = new double[N];
    auto *Atmp = new double[linePerProc[rank]];
    auto *y = new double[linePerProc[rank]];
    auto *tauY = new double[linePerProc[rank]];
    auto *temp = new double[N];
    double tau;
    double firstScalar;
    double secondScalar;
    double e = 1e-008;
    double bVecLenght;
    double yVecLenght;

    double result = 1;
    double prevResult = 1;
    bool diverge = false;
    int divergenceCount = 0;
    int convergentMatRepetition = 0;
    int cycleIterations = 0;

    FillMat(A, N, startPoints, linePerProc, rank);
    FillUVec(u, N, startPoints, linePerProc, rank);
    std::fill(currX, currX + linePerProc[rank], 0);
    std::fill(prevX, prevX + linePerProc[rank], 0);

    MatVecMul(A, u, temp, linePerProc, N, rank);
    MPI_Scatterv(temp, linePerProc, startPoints, MPI_DOUBLE, b, linePerProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    bVecLenght = sqrt(ScalProduct(b, b, linePerProc, rank));

    double startMeasureTime = MPI_Wtime();
    while (result >= e && convergentMatRepetition < CONVERGENCE) {
        if (result < e) {
            ++convergentMatRepetition;
        } else {
            convergentMatRepetition = 0;
        }

        MatVecMul(A, prevX, temp, linePerProc, N, rank); //y_n = Ax_n
        MPI_Scatterv(temp, linePerProc, startPoints, MPI_DOUBLE, Atmp, linePerProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        VecSub(Atmp, b, y, linePerProc, rank); //y_n = Ax_n - b
        yVecLenght = sqrt(ScalProduct(y, y, linePerProc, rank));
        MatVecMul(A, y, temp, linePerProc, N, rank); //Ay_n
        MPI_Scatterv(temp, linePerProc, startPoints, MPI_DOUBLE, Atmp, linePerProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        firstScalar = ScalProduct(y, Atmp, linePerProc, rank); //(y_n, Ay_n)
        secondScalar = ScalProduct(Atmp, Atmp, linePerProc, rank); //(Ay_n, Ay_n)
        tau = firstScalar / secondScalar;
        VecByNumMul(tau, y , tauY, linePerProc, rank);
        VecSub(prevX, tauY, currX, linePerProc, rank); //x_n+1 = x_n - t_ny_n
        result = yVecLenght / bVecLenght;

        if (prevResult < result) {
            divergenceCount++;
            if (divergenceCount > NONCONVERGENCE || result == INFINITY) {
                diverge = true;
                break;
            }
        }
        else {
            divergenceCount = 0;
        }
        prevResult = result;
        for (int i = 0; i < linePerProc[rank]; i++) {
            prevX[i] = currX[i];
        }
        cycleIterations++;
    }
    double endMeasureTime = MPI_Wtime();

    if (diverge) {
        std::cout << "Impossible task (╥﹏╥)" << std::endl << "Matrix is not convergent!" << std::endl;
        delete[](temp);
        delete[](tauY);
        delete[](y);
        delete[](Atmp);
        delete[](prevX);
        delete[](currX);
        delete[](b);
        delete[](fullU);
        delete[](u);
        delete[](A);
        MPI_Finalize();
        return 0;
    }

    MPI_Gatherv(currX, linePerProc[rank], MPI_DOUBLE, fullX, linePerProc, startPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    currX = fullX;
    MPI_Gatherv(u, linePerProc[rank], MPI_DOUBLE, fullU, linePerProc, startPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        PrintVec(fullU, N);
        std::cout << "u[]" << std::endl << std::endl;
        PrintVec(currX, N);
        std::cout << "x[]" << std::endl << std::endl;

        std::cout << "Amount of processes is: " << size << std::endl << std::endl;
        std::cout << "Amount of iterations: " << cycleIterations << std::endl;
        std::cout << "Total time is: " << endMeasureTime - startMeasureTime << "seconds" << std::endl;
    }
    delete[](temp);
    delete[](tauY);
    delete[](y);
    delete[](Atmp);
    delete[](prevX);
    delete[](currX);
    delete[](b);
    delete[](fullU);
    delete[](u);
    delete[](A);
    MPI_Finalize();
    return 0;
}
