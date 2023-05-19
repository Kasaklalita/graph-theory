from graph import Graph
from edge import Edge
from utils import InputType, print_help_info, print_success
import sys
from typing import List, Set
from graph_utils import kruskal, prim, boruvka
import time


def main():
    inputType: InputType = InputType.EDGE_LIST
    inputPath: str
    outputPath: str = ""
    key: str

    inputKeyExists = False
    multipleInputkeys = False
    helpKeyExists = False
    outputKeyExists = False

    algorithm_type: str = "-s"

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

        # Ключ для алгоритма
        if key == "-p" or key == "-k" or key == "-b" or key == "-s":
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

    # Матрица соотнесённого графа
    matrix = [[0 for j in range(len(g.adj_matrix))] for i in range(len(g.adj_matrix))]

    # Если граф ориентированный, работа идет с соотнесенным графом
    if g.directed:
        # Отзеркаливание матрицы для получения симметричной матрицы
        for i in range(len(g.adj_matrix)):
            for j in range(len(g.adj_matrix)):
                if g.adj_matrix[i][j] != 0:
                    # Если вершина i связана со вершиной j, то записываем связь в матрицу corr_matrix
                    matrix[i][j] = g.adj_matrix[i][j]
                    # также записываем связь в обратном направлении
                    matrix[j][i] = g.adj_matrix[i][j]
    else:
        # Просто копируем исходную матрицу
        for i in range(g.vertex_num):
            for j in range(g.vertex_num):
                matrix[i][j] = g.adj_matrix[i][j]

    # Создание итогового графа
    corr_graph = Graph(matrix)

    # Алгоритмы
    tree: List[Edge]

    match algorithm_type:
        case "-k":
            tree = kruskal(corr_graph)
            print_results(tree, outputPath)
        case "-p":
            tree = prim(corr_graph)
            print_results(tree, outputPath)
        case "-b":
            tree = boruvka(corr_graph)
            print_results(tree, outputPath)
        case _:
            # Алгоритм Краскала
            start_time = time.time()
            tree = kruskal(corr_graph)
            end_time = time.time()
            if outputKeyExists:
                fout = open(outputPath, "a")
                fout.write("Алгоритм Краскала\n")
                fout.write(
                    f"Длительность: {(end_time - start_time) * 1000} миллисекунд\n"
                )
                fout.close()
            else:
                print("Алгоритм Краскала")
                print(f"Длительность: {(end_time - start_time) * 1000} миллисекунд")
            print_results(tree, outputPath)

            # Алгоритм Прима
            start_time = time.time()
            tree = prim(corr_graph)
            end_time = time.time()
            if outputKeyExists:
                fout = open(outputPath, "a")
                fout.write("Алгоритм Прима\n")
                fout.write(
                    f"Длительность: {(end_time - start_time) * 1000} миллисекунд\n"
                )
                fout.close()
            else:
                print("Алгоритм Прима")
                print(f"Длительность: {(end_time - start_time) * 1000} миллисекунд")
            print_results(tree, outputPath)

            # Алгоритм Борувки
            start_time = time.time()
            tree = boruvka(corr_graph)
            end_time = time.time()
            if outputKeyExists:
                fout = open(outputPath, "a")
                fout.write("Алгоритм Борувки\n")
                fout.write(
                    f"Длительность: {(end_time - start_time) * 1000} миллисекунд\n"
                )
                fout.close()
            else:
                print("Алгоритм Борувки")
                print(f"Длительность: {(end_time - start_time) * 1000} миллисекунд")
            print_results(tree, outputPath)


def print_results(tree: List[Edge], output_path: str):
    if output_path == "":
        weight = 0
        print_success("Минимальное остовное дерево:")
        for edge in tree:
            print_success(f"({edge.__str__()})", end=" ")
            weight += edge.weight
        print_success(f"\nВес: {weight}")
    else:
        weight = 0
        fout = open(output_path, "a")
        fout.write("Минимальное остовное дерево:\n")
        for edge in tree:
            fout.write(f"({edge.__str__()}) ")
            weight += edge.weight
        fout.write(f"\nВес: {weight}\n")


if __name__ == "__main__":
    main()
