#include <iostream>
#include "mpi.h"

void printMat(double *matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++) {
            std::cout << matrix[i * columns + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    int Y = 0;
    int X = 1;
    const int dimensionsNum = 2; //количество измерений

    double *A;
    double *B;
    double *C;

    int dimSize[dimensionsNum] = {0}; //массив содержащий размер решётки
    int periods[dimensionsNum] = {0}; //зацикливание в поле сетки
    int coordinates[dimensionsNum]; //координаты
    MPI_Comm commGrid; //сеточный коммуниактор
    MPI_Comm commRow; //построчный коммуникатор
    MPI_Comm commColumn; //постолбцовый коммуникатор

    int N1 = atoi(argv[1]);
    int N2 = atoi(argv[2]);
    int N3 = atoi(argv[3]);

    if (argc == 6) {
        dimSize[Y] = atoi(argv[4]);
        dimSize[X] = atoi(argv[5]);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Dims_create(size, dimensionsNum, dimSize);
    ////Функция определения оптимальной конфигурации сетки
    //IN	size - общее число узлов в сетке(число процессов);
    //IN	dimensionsNum -	число измерений;
    //INOUT	dimSize - массив целого типа размерности dimensionsNum, в который помещается рекомендуемое число процессов вдоль каждого измерения


    MPI_Cart_create(MPI_COMM_WORLD, dimensionsNum, dimSize, periods, 0, &commGrid);
    ////Функция создания коммуникатора с декартовой топологией
    //создаёт коммуникатор решётки так чтобы процессы в рамках решетки могли коммуницироать со своими соседями
    //reorder сохраняет порядок rankов процессов в решетке согласно родительскому коммуникатору

    //IN	MPI_COMM_WORLD - родительский коммуникатор;
    //IN	dimensionsNum -	число измерений;
    //IN	dimSize - массив размера ndims, в котором задается число процессов вдоль каждого измерения;
    //IN	periods - логический массив размера ndims для задания граничных условий (true - периодические, false - непериодические);
    //IN	reorder(0)	- логическая переменная, указывает, производить перенумерацию процессов (true) или нет (false);
    //OUT	commGrid -	новый коммуникатор.


    MPI_Cart_coords(commGrid, rank, dimensionsNum, coordinates);
    ////Функция определения координат процесса по его идентификатору
    //фун-ция записывающая координаты процесса в решетке в зависимости от его ранка

    //IN	commGrid - коммуникатор с декартовой топологией;
    //IN	rank - идентификатор процесса;
    //IN	dimensionsNum - число измерений;
    //OUT	coordinates - координаты процесса в декартовой топологии.


    MPI_Comm_split(commGrid, coordinates[X], coordinates[Y], &commRow);
    ////Функция расщепления коммуникатора
    //разделение на построчные коммуникаторы

    //IN	commGrid - родительский коммуникатор;
    //IN	coordinates[X] -	признак подгруппы;
    //IN	coordinates[Y] - управление упорядочиванием;
    //OUT	commRow	- новый коммуникатор.


    MPI_Comm_split(commGrid, coordinates[Y], coordinates[X], &commColumn);
    ////Функция расщепления коммуникатора
    //разделение на постолбцовые коммуникаторы

    //IN	commGrid - родительский коммуникатор;
    //IN	coordinates[Y] -	признак подгруппы;
    //IN	coordinates[X] - управление упорядочиванием;
    //OUT	commColumn	- новый коммуникатор.


    if (rank == 0) {
        A = new double[N1 * N2];
        B = new double[N2 * N3];
        C = new double[N1 * N3];

        for (int i = 0; i < N1; i++) {
            for (int j = 0; j < N2; j++) {
                A[i * N2 + j] = i * N2 + j;
            }
        }
        for (int i = 0; i < N2; i++) {
            for (int j = 0; j < N3; j++) {
                B[i * N3 + j] = i * N3 + j;
            }
        }
    }

    int rowPieces = N1 / dimSize[X];
    int columnPieces = N3 / dimSize[Y];
    auto *partOfA = new double[rowPieces * N2];
    auto *partOfB = new double[N2 * columnPieces];
    auto *partOfC = new double[rowPieces * columnPieces];
    std::fill(partOfC, partOfC + rowPieces * columnPieces, 0);

    double startMeasureTime = MPI_Wtime();
    /// Distribute matrices
    if (coordinates[Y] == 0) {
        MPI_Scatter(A, rowPieces * N2, MPI_DOUBLE, partOfA, rowPieces * N2, MPI_DOUBLE, 0, commColumn);
        ////Функция распределения блоков данных по всем процессам группы
        //Рассылаем в ячейки слева для начала

        //IN	A - адрес начала размещения блоков распределяемых данных (используется только в процессе-отправителе root);
        //IN	rowPieces * N2 - число элементов, посылаемых каждому процессу;
        //IN	MPI_DOUBLE - тип посылаемых элементов;
        //OUT	partOfA	- адрес начала буфера приема;
        //IN	rowPieces * N2 - число получаемых элементов;
        //IN	MPI_DOUBLE - тип получаемых элементов;
        //IN	0 - номер процесса-отправителя;
        //IN	commColumn - коммуникатор.

    }
    MPI_Bcast(partOfA, rowPieces * N2, MPI_DOUBLE, 0, commRow);
    ////Широковещательная рассылка данных
    //Процесс с номером 0 рассылает сообщение из своего буфера передачи всем процессам области связи коммуникатора commRow
    //раздача в пределах своей строки

    //INOUT	partOfA - адрес начала расположения в памяти рассылаемых данных;
    //IN	rowPieces * N2 - число посылаемых элементов;
    //IN	MPI_DOUBLE - тип посылаемых элементов;
    //IN	root(0) - номер процесса-отправителя;
    //IN	commRow - коммуникатор.

    if (coordinates[X] == 0) {
        MPI_Datatype sendPieceType;
        //Производные типы MPI используются только в коммуникационных оперциях
        //Использование производного типа в функциях обмена сообщениями можно рассматривать как трафарет,
        //наложенный на область памяти, которая содержит передаваемое или принятое сообщение.

        MPI_Type_vector(N2, columnPieces, N3, MPI_DOUBLE, &sendPieceType);
        ////Конструктор типа MPI_Type_vector создает тип, элемент которого представляет собой несколько
        ///равноудаленных друг от друга блоков из одинакового числа смежных элементов базового типа

        //IN	N2 - число блоков;
        //IN	columnPieces - число элементов базового типа в каждом блоке;
        //IN	N3 - шаг между началами соседних блоков, измеренный числом элементов базового типа;
        //IN	MPI_DOUBLE - базовый тип данных;
        //OUT	sendPieceType - новый производный тип данных.


        MPI_Type_commit(&sendPieceType);
        ////Функция регистрирует созданный производный тип, только после регистрации новый тип
        ///может использоваться в коммуникационных операциях

        //INOUT sendPieceType - новый производный тип данных



        if (rank == 0) {
            for (int i = 0; i < N2; i++) {
                for (int j = 0; j < columnPieces; j++) {
                    partOfB[i * columnPieces + j] = B[i * N3 + j];
                }
            }
            //нулевой процесс копирует свой кусочек в цикле, так как сэндом он не сможет себе его отпарвить


            for (int i = 1; i < dimSize[Y]; i++) {
                MPI_Send(B + columnPieces * i, 1, sendPieceType, i, 1, commRow);
                ////Функция передачи сообщения
                //отправляем процессам каждому свой кусочек

                //IN B + columnPieces * i - адрес начала расположения пересылаемых данных;
                //IN count(1) - число пересылаемых элементов;
                //IN sendPieceType - тип посылаемых элементов;
                //IN i - номер процесса-получателя в группе, связанной с коммуникатором commRow;
                //IN tag(1) - идентификатор сообщения (аналог типа сообщения функций nread и nwrite PSE nCUBE2);
                //IN commRow - коммуникатор области связи.


            }
        } else {
            MPI_Recv(partOfB, N2 * columnPieces, MPI_DOUBLE, 0, 1, commRow, MPI_STATUS_IGNORE);
            ////Функция приема сообщения
            //ненулевые процессы получают свои кусочки

            //OUT	partOfB - адрес начала расположения принимаемого сообщения;
            //IN	N2 * columnPieces - максимальное число принимаемых элементов;
            //IN	MPI_DOUBLE - тип элементов принимаемого сообщения;
            //IN	source(0) - номер процесса-отправителя;
            //IN	tag(1) - идентификатор сообщения;
            //IN	commRow - коммуникатор области связи;
            //OUT	MPI_STATUS_IGNORE - атрибуты принятого сообщения.
        }



        MPI_Type_free(&sendPieceType);
        ////Функция уничтожает описатель производного типа
        //INOUT sendPieceType - уничтожаемый производный тип
    }



    MPI_Bcast(partOfB, N2 * columnPieces, MPI_DOUBLE, 0, commColumn);
    ////Широковещательная рассылка данных
    //Процесс с номером 0 рассылает сообщение из своего буфера передачи всем процессам области связи коммуникатора commColumn
    //раздача в пределах своего столбца

    //INOUT	partOfB - адрес начала расположения в памяти рассылаемых данных;
    //IN	N2 * columnPieces - число посылаемых элементов;
    //IN	MPI_DOUBLE - тип посылаемых элементов;
    //IN	root(0) - номер процесса-отправителя;
    //IN	commColumn - коммуникатор.


    /// Matrix multiplication
    for (int i = 0; i < rowPieces; i++) {
        for (int k = 0; k < N2; k++) {
            for (int j = 0; j < columnPieces; j++) {
                partOfC[i * columnPieces + j] += partOfA[i * N2 + k] * partOfB[k * columnPieces + j];
            }
        }
    }

    /// collect C
    MPI_Datatype receivePieceType;
    //Производные типы MPI используются только в коммуникационных оперциях
    //Использование производного типа в функциях обмена сообщениями можно рассматривать как трафарет,
    //наложенный на область памяти, которая содержит передаваемое или принятое сообщение.

    MPI_Type_vector(rowPieces, columnPieces, N3, MPI_DOUBLE, &receivePieceType);
    ////Конструктор типа MPI_Type_vector создает тип, элемент которого представляет собой несколько
    ///равноудаленных друг от друга блоков из одинакового числа смежных элементов базового типа

    //IN	rowPieces - число блоков;
    //IN	columnPieces - число элементов базового типа в каждом блоке;
    //IN	N3 - шаг между началами соседних блоков, измеренный числом элементов базового типа;
    //IN	MPI_DOUBLE - базовый тип данных;
    //OUT	receivePieceType - новый производный тип данных.


    MPI_Type_commit(&receivePieceType);
    ////Функция регистрирует созданный производный тип, только после регистрации новый тип
    ///может использоваться в коммуникационных операциях
    //наш тип "квадратик"

    //INOUT receivePieceType - новый производный тип данных

    int offsets[size];
    for (int procRank = 0; procRank < size; procRank++) {
        MPI_Cart_coords(commGrid, procRank, dimensionsNum, coordinates);
        ////Функция определения координат процесса по его идентификатору
        //распредление места в матрице C куда каждый процесс закинет свой "квадратик"
        //фун-ция записывающая координаты процесса в решетке в зависимости от его ранка

        //IN	commGrid - коммуникатор с декартовой топологией;
        //IN	procRank - идентификатор процесса;
        //IN	dimensionsNum - число измерений;
        //OUT	coordinates - координаты процесса в декартовой топологии.

        offsets[procRank] = coordinates[Y] * columnPieces + coordinates[X] * rowPieces * N3;
    }

    if (rank != 0) {
        MPI_Send(partOfC, rowPieces * columnPieces, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        ///Функция передачи сообщения
        //каждый ненулевой процесс отдает свой кусочек нулевому процессу

        //IN partOfC - адрес начала расположения пересылаемых данных;
        //IN rowPieces * columnPieces - число пересылаемых элементов;
        //IN MPI_DOUBLE - тип посылаемых элементов;
        //IN 0 - номер процесса-получателя в группе, связанной с коммуникатором commRow;
        //IN tag(1) - идентификатор сообщения (аналог типа сообщения функций nread и nwrite PSE nCUBE2);
        //IN MPI_COMM_WORLD - коммуникатор области связи.

    } else {
        for (int i = 0; i < rowPieces; i++) {
            for (int j = 0; j < columnPieces; j++) {
                C[i * N3 + j] = partOfC[i * columnPieces + j];
            }
            //копирует свой кусочек, так как нельзя сендить самому же себе
        }
        for (int i = 1; i < size; i++) {
            MPI_Recv(C + offsets[i], 1, receivePieceType, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ////Функция приема сообщения
            //нулевой процесс получает от всех ненулевых процессов свои кусочки

            //OUT	C + offsets[i] - адрес начала расположения принимаемого сообщения;
            //IN	1 - максимальное число принимаемых элементов;
            //IN	receivePieceType - тип элементов принимаемого сообщения;
            //IN	i - номер процесса-отправителя;
            //IN	tag(1) - идентификатор сообщения;
            //IN	MPI_COMM_WORLD - коммуникатор области связи;
            //OUT	MPI_STATUS_IGNORE - атрибуты принятого сообщения.
        }
    }


    MPI_Type_free(&receivePieceType);
    ////Функция уничтожает описатель производного типа
    //INOUT receivePieceType - уничтожаемый производный тип

    MPI_Comm_free(&commGrid);
    MPI_Comm_free(&commColumn);
    MPI_Comm_free(&commRow);
    ////Функция уничтожения коммуникатора
    //IN comm<Name> - уничтожаемый коммуникатор

    double endMeasureTime = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Total time is: " << endMeasureTime - startMeasureTime << "seconds" << std::endl;

        std::cout << std::endl << "A[" << N1 << "][" << N2 << "]" << std::endl;
        printMat(A, N1, N2);
        std::cout << std::endl << "B[" << N2 << "][" << N3 << "]" << std::endl;
        printMat(B, N2, N3);
        std::cout << std::endl << "C[" << N1 << "][" << N3 << "]" << std::endl;
        printMat(C, N1, N3);
    }

    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }
    free(partOfA);
    free(partOfB);
    free(partOfC);

    MPI_Finalize();
    return 0;
}