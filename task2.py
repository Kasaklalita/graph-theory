from graph import Graph
from vertex import Vertex
from utils import InputType, print_help_info
import sys
from typing import List, Set
from graph_utils import BFS


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
    # print(g.directed, g.edge_list.__len__(), g.adj_matrix, g.vertex_num)]]

    if g.directed:
        # Создание соотнесённого графа
        # Создаем матрицу соотнесенного графа, заполненную нулями
        corr_matrix = [
            [0 for j in range(len(g.adj_matrix))] for i in range(len(g.adj_matrix))
        ]

        # Проходим по каждой вершине графа и ее смежным вершинам
        for i in range(len(g.adj_matrix)):
            for j in range(len(g.adj_matrix)):
                if g.adj_matrix[i][j] == 1:
                    # Если вершина i связана со вершиной j, то записываем связь в матрицу corr_matrix
                    corr_matrix[i][j] = 1
                    # также записываем связь в обратном направлении
                    corr_matrix[j][i] = 1

        # Непосредственное создание
        corr_graph = Graph(corr_matrix)

        # Подсчёт компонент связности соотнесённого графа
        # Вектор посещённости
        visited: List[bool] = [False] * corr_graph.vertex_num
        # Cписок компонент связности соотнесенного графа
        components: List[Set[int]] = []

        # Обход в ширину каждой непосещенной на момент рассмотрения вершины
        for i in range(0, corr_graph.vertex_num):
            if not visited[i]:
                component: List[int] = []
                BFS(corr_graph, Vertex(i + 1), visited, component)
                res = set(component)
                components.append(res)

        # Теперь сильно связные компоненты
        sccs: List[Set[int]] = []
        for i in range(0, g.vertex_num):
            # Обнуление вектора посещённости
            visited = [False] * g.vertex_num
            pot_component: List[int] = []
            BFS(g, Vertex(i + 1), visited, pot_component)
            scc: Set[int] = {i + 1}

            for j in range(0, g.vertex_num):
                reverse_visited: List[bool] = [False] * g.vertex_num
                reverse_component: List[int] = []
                if j != i and visited[j]:
                    BFS(g, Vertex(j + 1), reverse_visited, reverse_component)
                    if (i + 1) in reverse_component:
                        scc.add(j + 1)

            if scc not in sccs:
                sccs.append(scc)

        if outputKeyExists:
            outputFile = open(outputPath, "w")
            if len(components) == 1:
                outputFile.write("Ориентированный граф связен.\n")
            else:
                outputFile.write(
                    f"Ориентированный граф не связен и содержит {len(components)} компонент связности\n"
                )
            outputFile.write(f"Компоненты связности: {components}\n")
            if len(sccs) != 1:
                outputFile.write(
                    f"Ориентированный граф слабо связен и содержит {len(sccs)} компонент сильной связности\n"
                )
            else:
                outputFile.write("Ориентированный граф сильно связен\n")
            outputFile.write(f"Компоненты сильной связности: {sccs}")

        else:
            if len(components) == 1:
                print("Ориентированный граф связен.")
            else:
                print(
                    f"Ориентированный граф не связен и содержит {len(components)} компонент связности"
                )
            print(f"Компоненты связности: {components}")
            if len(sccs) != 1:
                print(
                    f"Ориентированный граф слабо связен и содержит {len(sccs)} компонент сильной связности"
                )
            else:
                print("Ориентированный граф сильно связен")
            print(f"Компоненты сильной связности: {sccs}")

    else:
        # Подсчёт компонент связности соотнесённого графа
        # Вектор посещённости
        visited: List[bool] = [False] * g.vertex_num
        # Cписок компонент связности соотнесенного графа
        components: List[Set[int]] = []

        # Обход в ширину каждой непосещенной на момент рассмотрения вершины
        for i in range(0, g.vertex_num):
            if not visited[i]:
                component: List[int] = []
                BFS(g, Vertex(i + 1), visited, component)
                res = set(component)
                components.append(res)

        if outputKeyExists:
            outputFile = open(outputPath, "w")
            if len(components) == 1:
                outputFile.write("Граф связен.\n")
            else:
                outputFile.write(
                    f"Граф не связен и содержит {len(components)} компонент связности\n"
                )

            outputFile.write(f"Компоненты связности: {components}\n")

        else:
            if len(components) == 1:
                print("Граф связен.")
            else:
                print(
                    f"Граф не связен и содержит {len(components)} компонент связности"
                )

            print(f"Компоненты связности: {components}")


if __name__ == "__main__":
    main()
