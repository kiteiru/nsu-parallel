#include <iostream>
#include <math.h>
#include "mpi.h"

#define CONVERGENCE 5
#define NONCONVERGENCE 5

void PrintVec(double* vector, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

void MatVecMul(double* A, double* B, double* C, int N, int* linePerProc, int* startPoints, int rank) {
    auto* vector = new double[linePerProc[rank]];
    for (int i = 0; i < linePerProc[rank]; i++) {
        vector[i] = 0;
        for (int j = 0; j < N; j++) {
            vector[i] += A[i * N + j] * B[j];
        }
    }
    MPI_Allgatherv(vector, linePerProc[rank], MPI_DOUBLE, C, linePerProc, startPoints, MPI_DOUBLE, MPI_COMM_WORLD);
    delete[](vector);
}

double ScalProduct(double* A, double* B, int* linePerProc, int* startPoints, int rank) {
    double partialSum = 0;
    double C;
    for (int i = startPoints[rank]; i < startPoints[rank] + linePerProc[rank]; i++) {
        partialSum += A[i] * B[i];
    }
    MPI_Allreduce(&partialSum, &C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return C;
}

void VecByNumMul(double A, double* B, double* C, int N, int* linePerProc, int* startPoints, int rank) {
    auto* vector = new double[N];
    std::fill(vector, vector + N, 0);
    for (int i = startPoints[rank]; i < startPoints[rank] + linePerProc[rank]; i++) {
        vector[i] = A * B[i];
    }
    MPI_Allreduce(vector, C, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](vector);
}

void VecSub(double* A, double* B, double* C, int N, int* linePerProc, int* startPoints, int rank) {
    auto* vector = new double[N];
    std::fill(vector, vector + N, 0);
    for (int i = startPoints[rank]; i < startPoints[rank] + linePerProc[rank]; i++) {
        vector[i] = A[i] - B[i];
    }
    MPI_Allreduce(vector, C, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](vector);
}

void ColumnsDistribution(int* linePerProc, int* startPoints, int size, int N) {
    int quotient = N / size; //Количество столбцов каждому процессу
    int remainder = N % size; //Если столбцы не кратны кол-ву процессов
    std::fill(linePerProc, linePerProc + size, quotient);
    linePerProc[size - 1] += remainder;
    startPoints[0] = 0;
    for (int i = 1; i < size; ++i) {
        startPoints[i] = startPoints[i - 1] + linePerProc[i - 1];
    }
}

void FillU(double *u, int N) { //YEY
    for (int i = 0; i < N; i++) {
        u[i] = sin((2 * 3.14159 * i) / N);
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


    FillMat(A, N, startPoints, linePerProc, rank);
    FillU(u, N);

    MatVecMul(A, u, b, N, linePerProc, startPoints, rank);

    std::fill(currX, currX + N, 0); //x_n+1 = {0};
    std::fill(prevX, prevX + N, 0); //x_n = {0};

    bVecLenght = sqrt(ScalProduct(b, b, linePerProc, startPoints, rank));

    double startMeasureTime = MPI_Wtime();
    while ((result > e) && (convergentMatRepetition < CONVERGENCE)) {
        if (result < e) {
            convergentMatRepetition++;
        }
        else {
            convergentMatRepetition = 0;
        }

        MatVecMul(A, prevX, y, N, linePerProc, startPoints, rank); //y_n = Ax_n
        yVecLenght = sqrt(ScalProduct(y, y, linePerProc, startPoints, rank));
        VecSub(y, b, y, N, linePerProc, startPoints, rank); //y_n = Ax_n - b
        MatVecMul(A, y, Atmp, N, linePerProc, startPoints, rank); //Ay_n
        firstScalar = ScalProduct(y, Atmp, linePerProc, startPoints, rank); //(y_n, Ay_n)
        secondScalar = ScalProduct(Atmp, Atmp, linePerProc, startPoints, rank); //(Ay_n, Ay_n)
        tau = firstScalar / secondScalar;
        VecByNumMul(tau, y, tauY, N, linePerProc, startPoints, rank);
        VecSub(prevX, tauY, currX, N, linePerProc, startPoints, rank); //x_n+1 = x_n - t_ny_n
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
        for (int i = 0; i < linePerProc[rank]; i++) {
            prevX[i] = currX[i];
        }
        cycleIterations++;
    }
    double endMeasureTime = MPI_Wtime();

    if (diverge) {
        std::cout << "Impossible task (╥﹏╥)" << std::endl << "Matrix is not convergent!" << std::endl;
        delete[](u);
        delete[](tauY);
        delete[](Atmp);
        delete[](y);
        delete[](b);
        delete[](prevX);
        delete[](currX);
        delete[](A);
        delete[](startPoints);
        delete[](linePerProc);
        MPI_Finalize(); //Завершение работы MPI
        return 0;
    }

    if (rank == 0) {
        PrintVec(u, N);
        std::cout << " u[] " << std::endl << std::endl;
        PrintVec(currX, N);
        std::cout << " x[] " << std::endl << std::endl;

        std::cout << "Amount of processes is: " << size << std::endl << std::endl;
        std::cout << "Amount of iterations: " << cycleIterations + 1 << std::endl;
        std::cout << "Total time is: " << endMeasureTime - startMeasureTime << " seconds" << std::endl;
    }
    delete[](u);
    delete[](tauY);
    delete[](Atmp);
    delete[](y);
    delete[](b);
    delete[](prevX);
    delete[](currX);
    delete[](A);
    delete[](startPoints);
    delete[](linePerProc);
    MPI_Finalize(); //Завершение работы MPI
    return 0;
}
