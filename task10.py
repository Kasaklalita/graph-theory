from graph import Graph
from utils import InputType, print_help_info, print_fail, print_success, INF
import sys
from graph_utils import find_source_and_sink, ford_fulkerson


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

    source_vertex, sink_vertex = find_source_and_sink(g)
    flow = ford_fulkerson(g, source_vertex, sink_vertex)

    # print(max_flow, source_vertex, sink_vertex)

    if outputKeyExists:
        max_flow = 0
        for i in range(len(flow)):
            max_flow += flow[source_vertex][i]
        fout = open(outputPath, "w")
        fout.write(f"Максимальный поток от {source_vertex + 1} до {sink_vertex + 1}")
        for edge in g.edge_list:
            fout.write(
                f"{edge.a + 1} {edge.b + 1} {flow[edge.a][edge.b]} / {edge.weight}"
            )

    else:
        max_flow = 0
        for i in range(len(flow)):
            max_flow += flow[source_vertex][i]
        print_success(
            f"{max_flow} - максимальный поток от {source_vertex + 1} до {sink_vertex + 1}"
        )
        for edge in g.edge_list:
            print_success(
                f"{edge.a + 1} {edge.b + 1} {flow[edge.a][edge.b]}/{edge.weight}"
            )


if __name__ == "__main__":
    main()
