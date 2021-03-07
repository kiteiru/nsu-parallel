#include <iostream>
#include <math.h>
#include "mpi.h"

#define CONVERGENCE 5
#define NONCONVERGENCE 5

void PrintVector(double* vector, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << vector[i] << " ";
    }
}

void FillVectorWithZero(double* vector, int N) {
    for (int i = 0; i < N; i++) {
        vector[i] = 0;
    }
}

void MatrixAndVectorMultiplication(double* A, double* B, double* C, int N, int sizePerProcess) {
    double* vector = (double*)malloc(sizePerProcess * sizeof(double));
    for (int i = 0; i < sizePerProcess; i++) {
        vector[i] = 0;
        for (int j = 0; j < N; j++) {
            vector[i] += A[i * N + j] * B[j];
        }
    }
    MPI_Allgather(vector, sizePerProcess, MPI_DOUBLE, C, sizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);
    delete[](vector);
}

double ScalarVectorsMultiplication(double* A, double* B, int sizePerProcess, int rank) {
    double partialSum = 0;
    double C;
    for (int i = sizePerProcess * rank; i < sizePerProcess * rank + sizePerProcess; i++) {
        partialSum += A[i] * B[i];
    }
    MPI_Allreduce(&partialSum, &C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return C;
}

void ScalarAndVectorMultiplication(double A, double* B, double* C, int N, int sizePerProcess, int rank) {
    double* vector = (double*)malloc(N * sizeof(double));
    //std::fill(vector, vector + N, 0);
    FillVectorWithZero(vector, N);
    for (int i = sizePerProcess * rank; i < sizePerProcess * rank + sizePerProcess; i++) {
        vector[i] = A * B[i];
    }
    MPI_Allreduce(vector, C, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](vector);
}

void VectorsSubstruction(double* A, double* B, double* C, int N, int sizePerProcess, int rank) {
    double* vector = (double*)malloc(N * sizeof(double));
    //std::fill(vector, vector + N, 0);
    FillVectorWithZero(vector, N);
    for (int i = sizePerProcess * rank; i < sizePerProcess * rank + sizePerProcess; i++) {
        vector[i] = A[i] - B[i];
    }
    MPI_Allreduce(vector, C, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](vector);
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

double SquaresSum(double* v, int N) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

double EndCycleCriteria(double* A, double* x, double* b, double bVectorLenght, int N, int sizePerProcess, int rank) {
    double vector[N];
    MatrixAndVectorMultiplication(A, x, vector, N, sizePerProcess);
    VectorsSubstruction(vector, b, vector, N, sizePerProcess, rank);
    double result = SquaresSum(vector, N) / bVectorLenght;
    return result;
}

void PrintMatrixA(double* A, int N, int sizePerProcess) {
    for (int i = 0; i < sizePerProcess; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << " A[] " << std::endl << std::endl;;
}

void FillMatrixA(double *A, int N, int sizePerProcess, int rank) {
    for (int i = 0; i < sizePerProcess; i++) {
        for (int j = 0; j < N; j++) {
            if (i + (sizePerProcess * rank) == j) {
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
    int size, rank;
    MPI_Init(&argc, &argv); //Инициализация MPI
    MPI_Comm_size(MPI_COMM_WORLD, &size); //Получение числа процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //Получение номера процесса
    int sizePerProcess = N / size;
    auto* A = new double[sizePerProcess * N];
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

    FillMatrixA(A, N, sizePerProcess, rank);
    PrintMatrixA(A, N, sizePerProcess);

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            u[i] = sin((2 * 3.14159 * i) / N);
            std::cout << u[i] << " ";
        }
        std::cout << " u[]" << std::endl << std::endl;
    }

    MatrixAndVectorMultiplication(A, u, b, N, sizePerProcess);

    if (rank == 0) {
        PrintVector(b, N);
        std::cout << " b[] " << std::endl << std::endl;
    }

    bVectorLenght = SquaresSum(b, N);
    //std::fill(x, x + N, 0);//x_0 = {0};
    FillVectorWithZero(x, N); //x_0 = {0};


    double result = EndCycleCriteria(A, x, b, bVectorLenght, N, sizePerProcess, rank);

    int amountOfCycleIterations = 0;
    double startMeasureTime = MPI_Wtime();
    while ((result > e) && (convergentMatrixRepetition < CONVERGENCE)) {
        amountOfCycleIterations++;
        convergentMatrixRepetition = CheckOnRepitition(result, e, convergentMatrixRepetition);
        previousResult = result;

        MatrixAndVectorMultiplication(A, x, y, N, sizePerProcess); //y_n = Ax_n
        VectorsSubstruction(y, b, y, N, sizePerProcess, rank); //y_n = Ax_n - b
        MatrixAndVectorMultiplication(A, y, Ay, N, sizePerProcess); //Ay_n
        firstScalar = ScalarVectorsMultiplication(y, Ay, sizePerProcess, rank); //(y_n, Ay_n)
        secondScalar = ScalarVectorsMultiplication(Ay, Ay, sizePerProcess, rank);
        t = firstScalar / secondScalar;
        ScalarAndVectorMultiplication(t, y, ty, N, sizePerProcess, rank);
        VectorsSubstruction(x, ty, x, N, sizePerProcess, rank); // x_n+1 = x_n - t_ny_n

        result = EndCycleCriteria(A, x, b, bVectorLenght, N, sizePerProcess, rank);
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
    double endMeasureTime = MPI_Wtime();

    if (nonConvergation) {
        std::cout << "Impossible task! Matrix is not convergent!" << std::endl;
        delete[](u);
        delete[](ty);
        delete[](Ay);
        delete[](y);
        delete[](b);
        delete[](x);
        delete[](A);
        MPI_Finalize(); //Завершение работы MPI
        return 0;
    }

    if (rank == 0) {
        PrintVector(x, N);
        std::cout << " x[] " << std::endl << std::endl;

        std::cout << "Amount of processes is: " << size << std::endl << std::endl;
        std::cout << "Amount of iterations: " << amountOfCycleIterations + 1 << std::endl;
        std::cout << "Total time is: " << endMeasureTime - startMeasureTime << " seconds" << std::endl;
    }
    delete[](u);
    delete[](ty);
    delete[](Ay);
    delete[](y);
    delete[](b);
    delete[](x);
    delete[](A);
    MPI_Finalize(); //Завершение работы MPI
    return 0;
}
