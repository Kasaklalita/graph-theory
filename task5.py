from graph import Graph
from utils import InputType, print_help_info, print_fail, print_success
import sys
from graph_utils import dijkstra
from copy import deepcopy


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
    end_vertex: int = 0

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

        if key == "-d":
            end_vertex = int(args[i + 1])

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
    path_size, edges = dijkstra(start_vertex, end_vertex, g)

    if outputKeyExists:
        fout = open(outputPath, "w")
        if len(edges) == 0:
            fout.write(f"Пути от {start_vertex} до {end_vertex} не существует\n")
        else:
            fout.write(
                f"Длина кратчайшего пути от {start_vertex} до {end_vertex} равна {path_size}\n"
            )
            fout.write("Путь: ")
            fout.write([edge.info_as_bridge() for edge in edges].__str__())

    else:
        if len(edges) == 0:
            print_fail(f"Пути от {start_vertex} до {end_vertex} не существует")
        else:
            print_success(
                f"Длина кратчайшего пути от {start_vertex} до {end_vertex} равна {path_size}"
            )
            print_success("Путь: ", end="")
            print_success([edge.__str__() for edge in edges])


if __name__ == "__main__":
    main()
