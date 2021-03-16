#include <iostream>
#include <math.h>
#include "mpi.h"

#define CONVERGENCE 5
#define NONCONVERGENCE 5

void PrintVec(double* vector, int N) { //YEY
    for (int i = 0; i < N; i++) {
        std::cout << vector[i] << " ";
    }
}


void MatVecMul(const double *A, double *B,  double *C, int *linePerProc, int *startPoints, int N, int rank) {
    auto *tmpVec = new double[N];
    for (int i = 0; i < N; i++) {

        double mulSum = 0;
        for (int j = 0; j < linePerProc[rank]; j++) {
            mulSum += A[j * N + i] * B[j];
        }
        tmpVec[i] = mulSum;
    }
    MPI_Allreduce(tmpVec, C, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[](tmpVec);
}

double ScalProduct(const double *A, const double *B, int *linePerProc, int rank) { //YEY
    double partialSum = 0;
    double C;
    for (int i = 0; i < linePerProc[rank]; i++) {
        partialSum += A[i] * B[i];
    }
    MPI_Allreduce(&partialSum, &C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return C;
}

void VecByNumMul(double A, double *B, double *C, int *linePerProc, int rank) { //YEY
    for (int i = 0; i < linePerProc[rank]; i++) {
        C[i] = A * B[i];
    }
}

void VecSub(const double *A, const double *B, double *C, int *linePerProc, int rank) { //YEY
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
    int size, rank;
    MPI_Init(&argc, &argv); //Инициализация MPI
    MPI_Comm_size(MPI_COMM_WORLD, &size); //Получение числа процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //Получение номера процесса

    auto* linePerProc = new int[size];
    auto* startPoints = new int[size];
    ColumnsDistribution(linePerProc, startPoints, size, N);

    auto* A = new double[linePerProc[rank] * N];
    auto* u = new double[linePerProc[rank]];
    auto* fullU = new double[N];
    auto* b = new double[linePerProc[rank]];
    auto* currX = new double[linePerProc[rank]]; // xn+1
    auto* prevX = new double[linePerProc[rank]]; // xn
    auto* fullX = new double[N];
    auto* Atmp = new double[linePerProc[rank]];
    auto* y = new double[linePerProc[rank]];
    auto* tauY = new double[linePerProc[rank]];
    double tau;
    double firstScalar;
    double secondScalar;
    double e = 1e-008;
    double yVecLenght;
    double bVecLenght;

    double result = 1;
    double prevResult = 1;
    bool diverge = false;
    int divergenceCount = 0;
    int convergentMatRepetition = 0;
    int cycleIterations = 0;

    FillMat(A, N, startPoints, linePerProc, rank);
    FillUVec(u, N, startPoints, linePerProc, rank);
    std::fill(currX, currX + linePerProc[rank], 0);

    PrintVec(currX, linePerProc[rank]);

    std::fill(prevX, prevX + linePerProc[rank], 0);

    auto* temp = new double[N];

    MatVecMul(A, u, temp, linePerProc, startPoints, N, rank);
    MPI_Scatterv(temp, linePerProc, startPoints, MPI_DOUBLE, b, linePerProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    bVecLenght = sqrt(ScalProduct(b, b, linePerProc, rank));

    double startMeasureTime = MPI_Wtime();
    while ((result > e) && (convergentMatRepetition < CONVERGENCE)) {
        if (result < e) {
            convergentMatRepetition++;
        }
        else {
            convergentMatRepetition = 0;
        }

        MatVecMul(A, prevX, temp, linePerProc, startPoints, N, rank);
        MPI_Scatterv(temp, linePerProc, startPoints, MPI_DOUBLE, y, linePerProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        yVecLenght = sqrt(ScalProduct(y, y, linePerProc, rank));
        VecSub(y, b, y, linePerProc, rank);

        MatVecMul(A, y, temp, linePerProc, startPoints, N, rank);
        MPI_Scatterv(temp, linePerProc, startPoints, MPI_DOUBLE, Atmp, linePerProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        firstScalar = ScalProduct(y, Atmp, linePerProc, rank);
        secondScalar = ScalProduct(Atmp, Atmp, linePerProc, rank);
        tau = firstScalar / secondScalar;
        VecByNumMul(tau, y, tauY, linePerProc, rank);
        VecSub(prevX, tauY, currX, linePerProc, rank);

        PrintVec(currX, linePerProc[rank]);

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
        std::cout << "Impossible task! Matrix is not convergent!" << std::endl;
        delete[](prevX);
        delete[](Atmp);
        delete[](y);
        delete[](tauY);
        delete[](fullX);
        delete[](fullU);
        delete[](u);
        delete[](startPoints);
        delete[](linePerProc);
        delete[](currX);
        delete[](b);
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
        PrintVec(fullX, N);
        std::cout << "x[]" << std::endl << std::endl;

        std::cout << "Amount of processes is: " << size << std::endl << std::endl;
        std::cout << "Amount of iterations: " << cycleIterations << std::endl;
        std::cout << "Total time is: " << endMeasureTime - startMeasureTime << " seconds" << std::endl;
    }

    delete[](prevX);
    delete[](Atmp);
    delete[](y);
    delete[](tauY);
    delete[](fullU);
    delete[](u);
    delete[](startPoints);
    delete[](linePerProc);
    delete[](currX);
    delete[](b);
    delete[](A);
    MPI_Finalize();
    return 0;
}
