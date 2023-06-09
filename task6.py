from graph import Graph
from utils import InputType, print_help_info, print_fail, print_success
import sys
from graph_utils import (
    dijkstra_for_distances,
    bellman_ford_moore,
    levit,
    find_neg_cycle,
)


def main():
    inputType: InputType = InputType.EDGE_LIST
    inputPath: str
    outputPath: str = ""
    key: str

    inputKeyExists = False
    multipleInputkeys = False
    helpKeyExists = False
    outputKeyExists = False

    start_vertex: int = 0
    algorithm_type: str = "-d"

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

        if key == "-n":
            start_vertex = int(args[i + 1])

        if key == "-d" or key == "-b" or key == "-t":
            algorithm_type = key

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

    distances = []
    dijkstra_failed = False

    match algorithm_type:
        case "-d":
            if g.has_negative:
                dijkstra_failed = True
            if not dijkstra_failed:
                distances = dijkstra_for_distances(start_vertex, g)

        case "-b":
            distances = bellman_ford_moore(start_vertex, g)
        case "-t":
            distances = levit(start_vertex, g)

    result = []
    for i in range(len(distances)):
        result.append((start_vertex, i + 1, distances[i]))

    # path_size, edges = dijkstra_for_distances(start_vertex, g)

    if outputKeyExists:
        fout = open(outputPath, "w")
        fout.write("Кратчайшие расстояния:\n")
        for i in range(len(result)):
            fout.write(f"{result[i][0]} - {result[i][1]}: {result[i][2]}\n")

    else:
        print_success("Кратчайшие расстояния:")
        for i in range(len(result)):
            print_success(f"{result[i][0]} - {result[i][1]}: {result[i][2]}")


if __name__ == "__main__":
    main()
