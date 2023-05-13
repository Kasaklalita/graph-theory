from graph_1_task import Graph
from utils import InputType, print_help_info, print_fail, print_success, INF
import sys
from graph_utils import floyd_warshall
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

    degrees = [0] * g.vertex_num
    degrees_in = [0] * g.vertex_num
    degrees_out = [0] * g.vertex_num

    if not g.directed:
        for i in range(g.vertex_num):
            for j in range(g.vertex_num):
                if g.adj_matrix[i][j] != 0 and g.adj_matrix[i][j] != INF:
                    degrees[i] += 1
    else:
        for i in range(g.vertex_num):
            for j in range(g.vertex_num):
                if g.adj_matrix[i][j] != 0 and g.adj_matrix[i][j] != INF:
                    degrees_in[i] += 1
                if g.adj_matrix[j][i] != 0 and g.adj_matrix[i][j] != INF:
                    degrees_out[i] -= -1

    distances = floyd_warshall(g)

    diameter = 0
    radius = INF
    eccentricities = []

    # Нахождение минимумов и максимумов в матрице
    for i in range(len(distances)):
        # Нахождение очередного эксцентриситета
        e = max(distances[i])

        # Установка минимума и максимума
        diameter = max(diameter, e)
        radius = min(radius, e)

        # Добавление эксцентриситета в массив
        eccentricities.append(e)

    center_vertices = []
    peripheral_vertices = []
    for i in range(len(eccentricities)):
        if eccentricities[i] == diameter:
            peripheral_vertices.append(i)
        if eccentricities[i] == radius:
            center_vertices.append(i)

    if outputKeyExists:
        fout = open(outputPath, "w")
        if g.directed:
            fout.write(f"Степени исхода: {[degree_in for degree_in in degrees_in]}\n")
            fout.write(
                f"Степени захода: {[degree_out for degree_out in degrees_out]}\n"
            )
        else:
            fout.write(f"Вывод степеней вершин: {[degree for degree in degrees]}\n")

        fout.write("Матрица расстояний:\n")
        for i in range(len(distances)):
            for j in range(len(distances)):
                if distances[i][j] == INF:
                    fout.write("INF\t")
                else:
                    fout.write(f"{distances[i][j]}\t")
            fout.write("\n")

        if diameter != INF:
            fout.write(f"Эксцентриситеты: {[e for e in eccentricities]}\n")

        if not g.directed and diameter != INF:
            fout.write(f"Диаметр: {diameter}\n")
            fout.write(f"Радиус: {radius}\n")
            fout.write(f"Центральные вершины: {[v for v in center_vertices]}\n")
            fout.write(f"Периферийные вершины: {[v for v in peripheral_vertices]}\n")
    else:
        if g.directed:
            print(f"Степени исхода: {[degree_in for degree_in in degrees_in]}")
            print(f"Степени захода: {[degree_out for degree_out in degrees_out]}")
        else:
            print(f"Вывод степеней вершин: {[degree for degree in degrees]}")

        print("Матрица расстояний:\n")
        for i in range(len(distances)):
            for j in range(len(distances)):
                if distances[i][j] == INF:
                    print("∞\t", end="")
                else:
                    print(f"{distances[i][j]}\t", end="")
            print("")

        if diameter != INF:
            print(f"Эксцентриситеты: {[e for e in eccentricities]}")

        if not g.directed and diameter != INF:
            print(f"Диаметр: {diameter}")
            print(f"Радиус: {radius}")
            print(f"Центральные вершины: {[v + 1 for v in center_vertices]}")
            print(f"Периферийные вершины: {[v + 1 for v in peripheral_vertices]}")


if __name__ == "__main__":
    main()
