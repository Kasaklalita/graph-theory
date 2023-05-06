from typing import List
from utils import split_row, InputType, bcolors
from vertex import Vertex
from edge import Edge


class Graph:
    # bool directed = False
    # edge_list = Edge[]
    # ajd_matrix = int[[]]
    # int vertex_num

    @property
    def directed(self):
        return self.__directed

    @property
    def edge_list(self):
        return self.__edge_list

    @property
    def adj_matrix(self):
        return self.__adj_matrix

    @property
    def vertex_num(self):
        return self.__vertex_num

    # @directed.setter
    # def directed(self, directed):
    #     self.directed = directed

    # Определение веса ребра, связывающего вершины
    def weight(self, i: Vertex, j: Vertex) -> int:
        if self.edge_in_list(i, j):
            return self.get_edge(i, j).weight
        return 0

    # Список ребёр, инцидентных данной вершине
    def list_of_edges_by_vertex(self, v: Vertex) -> List[Edge]:
        res = []
        for edge in self.edge_list:
            if edge.a == v:
                res.append(edge)
        return res

    # Проверка наличия ребра с такими вершинами
    def edge_in_list(self, i: Vertex, j: Vertex) -> bool:
        for edge in self.__edge_list:
            if edge.a == i and edge.b == j:
                return True
        return False

    # Вывод информации о всех рёбрах графа
    def print_edjes_list(self) -> None:
        print("--- Edges List ---")
        for i in range(0, len(self.edge_list)):
            print(self.edge_list[i])

    # Получение списка вершин, смежных заданной вершине
    def adjacency_list(self, v: Vertex) -> List[Vertex]:
        res: List[Vertex] = []
        for i in range(0, self.vertex_num):
            if self.adj_matrix[v.number][i] != 0:
                res.append(Vertex(i + 1))
        return res

    # Вывод информации о матрице смежности графа
    def print_adjacency_matrix(self):
        print("Adjacency matrix:")
        for i in range(0, len(self.adj_matrix)):
            for j in range(0, len(self.adj_matrix)):
                print(self.__adj_matrix[i][j], end=" ")
            print("")

    # Получение ребра с заданными вершинами из списка
    def get_edge(self, i: Vertex, j: Vertex) -> Edge:
        for edge in self.edge_list:
            if edge.a == i and edge.b == j:
                return edge
        return Edge(-1, -1, -1)

    # Удаление ребра из списка рёбер
    def delete_edge(self, edge: Edge):
        self.__edge_list.pop(self.__edge_list.index(edge))

    # Добавление ребра в список рёбер
    def add_edge(self, edge: Edge):
        self.__edge_list.append(edge)

    def __init__(self, *args):
        self.__directed: bool = False
        self.__edge_list: List[Edge] = []
        self.__adj_matrix: List[List[int]] = []
        self.__vertex_num: int = 0

        # Создание через путь к файлу и тип ввода
        if len(args) == 2:
            path: str = args[0]
            input_type: InputType = args[1]

            row_number = 1
            try:
                fin = open(path, "r")
            except IOError:
                print(bcolors.FAIL + "Такой файл не найден" + bcolors.ENDC)
                quit()

            match input_type:
                # Матрица смежности
                case InputType.ADJ_MATRIX:
                    while True:
                        # Считывание строки матрицы
                        new_line = fin.readline()
                        if not new_line:
                            break

                        # Разбиение строки на ячейки
                        values = split_row(new_line)
                        if len(values) == 0:
                            continue
                        self.__vertex_num = len(values)

                        # Анализ каждой строки ячейки
                        for i in range(0, len(values)):
                            if values[i] != 0:
                                # Создание нового ребра
                                new_edge = Edge(row_number, i + 1, values[i])
                                # Проверка на уникальность
                                if not self.edge_in_list(new_edge.a, new_edge.b):
                                    self.edge_list.append(new_edge)
                        row_number += 1

                case InputType.ADJ_LIST:
                    while True:
                        # Прохождение по всем строкам
                        new_line = fin.readline()
                        if not new_line:
                            break

                        # Разбиение строки на ячейки
                        values = split_row(new_line)
                        for i in range(0, len(values)):
                            new_edge = Edge(row_number, values[i], 1)
                            if not self.edge_in_list(new_edge.a, new_edge.b):
                                self.edge_list.append(new_edge)
                        # Переход к следующей вершине
                        row_number += 1
                        self.__vertex_num += 1
                case InputType.EDGE_LIST:
                    vertex_list: List[Vertex] = []
                    while True:
                        # Чтение очередного ребра
                        new_line = fin.readline()
                        if not new_line:
                            break

                        # Разбиение строки на значения
                        values = split_row(new_line)
                        if len(values) == 0:
                            continue

                        weight = 0

                        # Создание нового ребра
                        if len(values) == 2:
                            weight = 1
                        else:
                            weight = values[2]
                        new_edge = Edge(values[0], values[1], weight)

                        # Проверка ребра на уникальность
                        if not self.edge_in_list(new_edge.a, new_edge.b):
                            self.__edge_list.append(new_edge)

                            # Добавление уникальных вершин в список вершин
                            if new_edge.a not in vertex_list:
                                vertex_list.append(new_edge.a)
                            if new_edge.b not in vertex_list:
                                vertex_list.append(new_edge.b)
                    self.__vertex_num = len(vertex_list)

            # Создание матрицы смежности
            for i in range(0, self.__vertex_num):
                self.__adj_matrix.append([0] * self.__vertex_num)
                for j in range(0, self.__vertex_num):
                    if not (self.edge_in_list(Vertex(i + 1), Vertex(j + 1))):
                        self.__adj_matrix[i][j] = 0
                    else:
                        self.__adj_matrix[i][j] = 1

            # Проверка ориентированности графа
            is_symmetrical = True
            for i in range(0, self.__vertex_num):
                for j in range(i + 1, self.__vertex_num):
                    if self.adj_matrix[i][j] != self.adj_matrix[j][i]:
                        is_symmetrical = False
                        break

            # Определение ориентированности графа
            self.__directed = not is_symmetrical

        # Создание через матрицу смежности
        elif len(args) == 1:
            matrix: List[List[int]] = args[0]
            self.__vertex_num = len(matrix)
            self.__adj_matrix = []

            # Заполнение матрицы и списка рёбер
            for i in range(0, self.vertex_num):
                self.__adj_matrix.append([0] * self.vertex_num)
                for j in range(0, self.vertex_num):
                    self.__adj_matrix[i][j] = matrix[i][j]
                    if matrix[i][j] != 0:
                        self.__edge_list.append(Edge(i + 1, j + 1, matrix[i][j]))

            # Проверика ориентированности графа
            is_symmetrical = True
            for i in range(0, self.vertex_num):
                for j in range(i + 1, self.vertex_num):
                    if self.__adj_matrix[i][j] != self.__adj_matrix[j][i]:
                        is_symmetrical = False
                        break

            self.__directed = not is_symmetrical
