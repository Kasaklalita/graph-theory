from enum import Enum
from typing import List


def split_row(line: str):
    ints = line.split(" ")
    return [int(entity) for entity in ints]


class InputType(Enum):
    ADJ_MATRIX = 1
    ADJ_LIST = 2
    EDGE_LIST = 3


class Vertex:
    def __init__(self, number: int):
        self.number = number

    def __eq__(self, other):
        return self.number == other.number

    def __lt__(self, other):
        return self.number < other.number

    def __gt__(self, other):
        return self.number > other.number

    def __str__(self) -> str:
        return f"{self.number}"


class Edge:
    def __init__(self, num_a: int, num_b: int, weight: int):
        self.a = Vertex(num_a)
        self.b = Vertex(num_b)
        self.weight = weight

    def print_edge(self) -> str:
        print(f"{self.a.number} {self.b.number} {self.weight}")
        return f"{self.a.number} {self.b.number} {self.weight}"

    def contains(self, v: Vertex) -> bool:
        return self.a == v or self.b == v

    def __str__(self) -> str:
        return f"{self.a.number} {self.b.number} {self.weight}"


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

    def weight(self, i: Vertex, j: Vertex) -> int:
        if self.edge_in_list(i, j):
            pass
        return 0

    def list_of_edges(self):
        return self.edge_list

    def list_of_edges_by_vertex(self, v: Vertex):
        res = []
        for edge in self.edge_list:
            if edge.a == v:
                res.append(edge)
        return res

    def edge_in_list(self, i: Vertex, j: Vertex) -> bool:
        for edge in self.__edge_list:
            if edge.a == i and edge.b == j:
                return True
        return False
    
    def print_edjes_list(self):
        print("--- Edges List ---")
        for 

    def is_directed(self):
        return self.directed

    # Получение матрицы смежности графа
    def adjacency_matrix(self):
        return self.adj_matrix

    # Получение списка вершин, смежных заданной вершине
    def adjacency_list(self, v: Vertex) -> List[Vertex]:
        res: List[Vertex] = []
        for i in range(0, self.vertex_num):
            if self.adj_matrix[v.number][i] != 0:
                res.append(Vertex(i + 1))
        return res

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

    def __init__(self, path: str, input_type: InputType) -> None:
        self.__directed: bool = False
        self.__edge_list: List[Edge] = []
        self.__adj_matrix: List[List[int]] = [[]]
        self.__vertex_num: int = 0

        row_number = 1
        fin = open(path, "r")

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

        #

        # Создание матрицы смежности
        self.__adj_matrix: List[List[int]] = [
            [0] * self.vertex_num
        ] * self.vertex_num
        # self.print_adjacency_matrix()

        for i in range(0, self.vertex_num):
            for j in range(0, self.vertex_num):
                print(i, j)
                if not self.edge_in_list(Vertex(i + 1), Vertex(j + 1)):
                    self.__adj_matrix[i][j] = 0
                else:
                    self.__adj_matrix[i][j] = self.get_edge(
                        Vertex(i + 1), Vertex(j + 1)
                    ).weight
