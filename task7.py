from graph import Graph
from utils import InputType, print_help_info, print_fail, print_success, INF
import sys
from graph_utils import find_neg_cycle, johnson


def main():
    inputType: InputType = InputType.EDGE_LIST
    inputPath: str
    outputPath: str = ""
    key: str

    inputKeyExists = False
    multipleInputkeys = False
    helpKeyExists = False
    outputKeyExists = False

    args = sys.argv[1:]
    for i in range(0, len(args)):
        key = args[i]
        # Если несколько ключей, то ошибка
        if (key == "-e" or key == "-m" or key == "-l") and inputKeyExists:
            multipleInputkeys = True
        # Первый ключ для ввода найден
        elif key == "-e" or key == "-m" or key == "-l":
            inputKeyExists = True

        # Если ключ помощи, то все остальные не нужны
        if key == "-h":
            helpKeyExists = True
            break

        # Ключ для вывода в файл
        if key == "-o":
            outputKeyExists = True
            outputPath = args[i + 1]

    # Вывод помощи
    if helpKeyExists:
        print_help_info()
        return 0

    # Проверки на ошибки
    if not inputKeyExists:
        print("Ошибка. Нет ключа для ввода данных")
    if multipleInputkeys:
        print("Ошибка. Несколько ключей для ввода данных")

    key = args[0]
    match key:
        case "-e":
            inputType = InputType.EDGE_LIST
        case "-m":
            inputType = InputType.ADJ_MATRIX
        case "-l":
            inputType = InputType.ADJ_LIST

    inputPath = args[1]

    g = Graph(inputPath, inputType)

    if g.has_negative and find_neg_cycle(g):
        if outputKeyExists:
            fout = open(outputPath, "w")
            fout.write("В графе есть отрицательный цикл\n")
            return
        else:
            print_fail("В графе есть отрицательный цикл")
            return

    if g.has_negative:
        if outputKeyExists:
            fout = open(outputPath, "w")
            fout.write("В графе есть рёбра отрицительного веса\n")
        else:
            print_fail("В графе есть рёбра отрицательного веса")
    else:
        if outputKeyExists:
            fout = open(outputPath, "w")
            fout.write("В графе нет рёбер отрицательного веса\n")
        else:
            print_success("В графе нет ребёр отрицательного веса")

    dist = johnson(g)

    if outputKeyExists:
        fout = open(outputPath, "w")
        fout.write("Кратчайшие пути:")
        for i in range(len(dist)):
            for j in range(len(dist)):
                if i != j and dist[i][j] < INF - 100:
                    fout.write(f"{i + 1} - {j + 1}: {dist[i][j]}\n")

    else:
        print_success("Кратчайшие пути:")
        for i in range(len(dist)):
            for j in range(len(dist)):
                if i != j and dist[i][j] < INF - 100:
                    print_success(f"{i + 1} - {j + 1}: {dist[i][j]}")


if __name__ == "__main__":
    main()
