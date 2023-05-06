from graph import Graph
from vertex import Vertex
from utils import InputType, print_help_info
import sys
from typing import List, Set
from graph_utils import BFS, find_joints, find_bridges


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

    # Матрица соотнесённого графа
    corr_matrix = [
        [0 for j in range(len(g.adj_matrix))] for i in range(len(g.adj_matrix))
    ]

    # Если граф ориентированный, работа идет с соотнесенным графом
    if g.directed:
        # Проходим по каждой вершине графа и ее смежным вершинам
        for i in range(len(g.adj_matrix)):
            for j in range(len(g.adj_matrix)):
                if g.adj_matrix[i][j] == 1:
                    # Если вершина i связана со вершиной j, то записываем связь в матрицу corr_matrix
                    corr_matrix[i][j] = 1
                    # также записываем связь в обратном направлении
                    corr_matrix[j][i] = 1
    else:
        # Просто копируем исходную матрицу
        for i in range(g.vertex_num):
            for j in range(g.vertex_num):
                corr_matrix[i][j] = g.adj_matrix[i][j]

    # Создание итогового графа
    corr_graph = Graph(corr_matrix)

    # Ищем шарниры и мосты
    joints = find_joints(corr_graph)
    bridges = find_bridges(corr_graph, joints)

    if outputKeyExists:
        fout = open(outputPath, "w")
        fout.write(f"Мосты: {[bridge.info_as_bridge() for bridge in bridges]}\n")
        fout.write(f"Шарниры: {[joint for joint in joints]}\n")
    else:
        print(f"Мосты: {[bridge.info_as_bridge() for bridge in bridges]}")
        print(f"Шарниры: {[joint for joint in joints]}")

    # if outputKeyExists:
    #     outputFile = open(outputPath, "w")
    #     if len(components) == 1:
    #         outputFile.write("Ориентированный граф связен.\n")
    #     else:
    #         outputFile.write(
    #             f"Ориентированный граф не связен и содержит {len(components)} компонент связности\n"
    #         )
    #     outputFile.write(f"Компоненты связности: {components}\n")
    #     if len(sccs) != 1:
    #         outputFile.write(
    #             f"Ориентированный граф слабо связен и содержит {len(sccs)} компонент сильной связности\n"
    #         )
    #     else:
    #         outputFile.write("Ориентированный граф сильно связен\n")
    #     outputFile.write(f"Компоненты сильной связности: {sccs}")

    # else:
    #     if len(components) == 1:
    #         print("Ориентированный граф связен.")
    #     else:
    #         print(
    #             f"Ориентированный граф не связен и содержит {len(components)} компонент связности"
    #         )
    #     print(f"Компоненты связности: {components}")
    #     if len(sccs) != 1:
    #         print(
    #             f"Ориентированный граф слабо связен и содержит {len(sccs)} компонент сильной связности"
    #         )
    #     else:
    #         print("Ориентированный граф сильно связен")
    #     print(f"Компоненты сильной связности: {sccs}")

    # else:
    #     # Подсчёт компонент связности соотнесённого графа
    #     # Вектор посещённости
    #     visited: List[bool] = [False] * g.vertex_num
    #     # Cписок компонент связности соотнесенного графа
    #     components: List[Set[int]] = []

    #     # Обход в ширину каждой непосещенной на момент рассмотрения вершины
    #     for i in range(0, g.vertex_num):
    #         if not visited[i]:
    #             component: List[int] = []
    #             BFS(g, Vertex(i + 1), visited, component)
    #             res = set(component)
    #             components.append(res)

    #     if outputKeyExists:
    #         outputFile = open(outputPath, "w")
    #         if len(components) == 1:
    #             outputFile.write("Граф связен.\n")
    #         else:
    #             outputFile.write(
    #                 f"Граф не связен и содержит {len(components)} компонент связности\n"
    #             )

    #         outputFile.write(f"Компоненты связности: {components}\n")

    #     else:
    #         if len(components) == 1:
    #             print("Граф связен.")
    #         else:
    #             print(
    #                 f"Граф не связен и содержит {len(components)} компонент связности"
    #             )

    #         print(f"Компоненты связности: {components}")


if __name__ == "__main__":
    main()
