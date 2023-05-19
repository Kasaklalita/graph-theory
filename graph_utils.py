from graph import Graph
from typing import List, Set, Tuple
from edge import Edge
from functools import cmp_to_key
from utils import INF
from copy import deepcopy
from collections import deque


def floyd_warshall(g: Graph) -> List[List[int]]:
    distances_matrix = deepcopy(g.adj_matrix)

    for i in range(len(distances_matrix)):
        for j in range(len(distances_matrix)):
            for k in range(len(distances_matrix)):
                if (
                    distances_matrix[j][k]
                    > distances_matrix[j][i] + distances_matrix[i][k]
                ):
                    distances_matrix[j][k] = (
                        distances_matrix[j][i] + distances_matrix[i][k]
                    )
    return distances_matrix


# Поиск в ширину
def BFS(g: Graph, start: int, visited: List[bool], container: List[int]):
    queue: List[int] = []
    queue.append(start)

    visited[start - 1] = True

    while len(queue) > 0:
        y = queue.pop()
        container.append(y)

        for edge in g.list_of_edges_by_vertex(y):
            if not visited[edge.b - 1]:
                queue.append(edge.b)
                visited[edge.b - 1] = True


def corresponding_matrix(matrix: List[List[int]], directed: bool = True):
    corr_matrix = [[0] * len(matrix)] * len(matrix)
    n = len(matrix)
    # for i in range(len(matrix)):
    #     for j in range(i, len(matrix)):
    #         if matrix[i][j] != 0:
    #             corr_matrix[i][j] = matrix[i][j]
    #             corr_matrix[j][i] = matrix[i][j]
    #         elif matrix[i][j] == 0:
    #             corr_matrix[i][j] = 0
    #             corr_matrix[j][i] = 0
    # return corr_matrix
    for i in range(n):
        for j in range(i + 1, n):
            ij, ji = matrix[i][j], matrix[j][i]
            print(ij, ji)
            if ji == 0:
                corr_matrix[j][i] = 1
            elif ij == 0:
                corr_matrix[i][j] = ji
    return corr_matrix


# Подсчёт компонент связности
def connectivity_components_count(graph: Graph):
    # Поиск в ширину для подсчёта компонент связности
    def BFS(g: Graph, start: int, visited: List[int]):
        queue: List[int] = []  # Очередь
        queue.append(start)

        # Добавление стартовой вершины в список посещённх
        visited.append(start)

        # Пока очередь не пуста
        while len(queue) != 0:
            # Получение крайнего элемента в очереди
            y = queue.pop()

            # Идём по всем рёбрам, содержащим вершину y
            for edge in g.list_of_edges_by_vertex(y):
                # Если другая вершина не посещена, то добавляем её в очередь
                if edge.b not in visited:
                    queue.append(edge.b)
                    visited.append(edge.b)

    visited: List[int] = []  # Список посещённых вершин
    result = 0  # Количество компонент связности
    for i in range(0, graph.vertex_num):
        if (i + 1) not in visited:
            BFS(graph, i, visited)
            result += 1
    return result


# Поиск шарниров в графе
def find_joints(graph: Graph):
    # Господи Боже, как же я намучился с этой функцией...
    visited = [False] * graph.vertex_num
    disc_time = [float("inf")] * graph.vertex_num
    low = [float("inf")] * graph.vertex_num
    parent = [-1] * graph.vertex_num
    is_articulation = [False] * graph.vertex_num
    time = 0

    def dfs(u):
        nonlocal time
        child_count = 0  # Количество потомков
        visited[u] = True  # Вершина посещена
        disc_time[u] = low[u] = time
        time += 1  # Увеличиваем время выхода

        for v in range(graph.vertex_num):
            if graph.adj_matrix[u][v]:
                if not visited[v]:
                    parent[v] = u
                    child_count += 1
                    dfs(v)

                    low[u] = min(low[u], low[v])

                    if parent[u] == -1 and child_count > 1:
                        is_articulation[u] = True

                    if parent[u] != -1 and low[v] >= disc_time[u]:
                        is_articulation[u] = True

                elif v != parent[u]:
                    low[u] = min(low[u], disc_time[v])

    for i in range(graph.vertex_num):
        if not visited[i]:
            dfs(i)

    return [i for i, is_art in enumerate(is_articulation) if is_art]


# Поиск мостов в графе СТАРАЯ ВЕРСИЯ
# def find_bridges(graph: Graph, joints: Set[int]) -> Set[Edge]:
#     result: Set[Edge] = set()  # Результат

#     # Идём по всем шарнирам
#     for joint in joints:
#         current_vertex = joint
#         # Список всех рёбер, исходящих из вершины
#         edge_list = graph.list_of_edges_by_vertex(current_vertex)
#         # Подсчёт компонент в исходном графе
#         prev_connectivity_components_count = connectivity_components_count(
#             graph
#         )

#         # Идём по всем рёбрам вершины
#         for edge in edge_list:
#             # Отзеркаливаем ребро, чтобы удалить и его
#             mirrored_edge = Edge(edge.b, edge.a, edge.weight)
#             # Удаление обоих рёбер
#             graph.delete_edge(edge)
#             graph.delete_edge(mirrored_edge)
#             # Считаем компоненты связности
#             current_connectivity_components_number = (
#                 connectivity_components_count(graph)
#             )
#             # Если число компонент увеличилось и это ребро ещё не смотрели
#             if (
#                 current_connectivity_components_number
#                 > prev_connectivity_components_count
#             ) and mirrored_edge not in result:
#                 result.add(edge)

#             # Восстанавливаем рёбра
#             graph.add_edge(edge)
#             graph.add_edge(mirrored_edge)

#     return result


# Нахождение мостов
def find_bridges(graph: Graph):
    # Инициализация переменных
    adj_matrix = graph.adj_matrix
    n = len(adj_matrix)
    bridges: List[Edge] = []
    visited = [False] * n
    entry_time = [0] * n
    exit_time = [0] * n
    time = 0

    def dfs(node, parent):
        # Обход в глубину
        nonlocal time
        visited[node] = True
        entry_time[node] = time
        exit_time[node] = time
        time += 1
        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and neighbor != parent:
                # Проверяем соседей текущей вершины
                if not visited[neighbor]:
                    # Обход не посещенного соседа
                    dfs(neighbor, node)
                    # Пересчет времени выхода из вершины
                    exit_time[node] = min(exit_time[node], exit_time[neighbor])
                    # Проверка наличия моста
                    if entry_time[node] < exit_time[neighbor]:
                        bridges.append(
                            Edge(
                                node,
                                neighbor,
                                graph.adj_matrix[node][neighbor],
                            )
                        )
                else:
                    # Обновление времени выхода из вершины
                    exit_time[node] = min(exit_time[node], entry_time[neighbor])

    for node in range(n):
        # Начинаем обход с каждой вершины
        if not visited[node]:
            dfs(node, None)

    return bridges


# # Компаратор для сортировки ребер по весу
# def edge_comparator(a: Edge, b: Edge) -> bool:
#     return (
#         a.weight < b.weight
#         or (a.weight == b.weight and a.a < b.a)
#         or (a.weight == b.weight and a.a == b.a and a.b < b.b)
#     )


# # Поиск вершины в списке множеств. Возвращает позицию множества с найденной вершиной
# def vertex_search(vertex_list: List[Set[int]], number: int) -> int:
#     for i in range(0, len(vertex_list)):
#         if number in vertex_list[i]:
#             return i
#     return -1


# def kruskal(g: Graph) -> Set[Edge]:
#     parent = dict()
#     rank = dict()

#     def make_set(vertice: int):
#         parent[vertice] = vertice
#         rank[vertice] = 0

#     def find(vertice: int):
#         if parent[vertice] != vertice:
#             parent[vertice] = find(parent[vertice])
#         return parent[vertice]

#     def union(vertice1: int, vertice2: int):
#         root1 = find(vertice1)
#         root2 = find(vertice2)
#         if root1 != root2:
#             if rank[root1] > rank[root2]:
#                 parent[root2] = root1
#         else:
#             parent[root1] = root2
#         if rank[root1] == rank[root2]:
#             rank[root2] += 1

#     # for vertice in graph["vertices"]:
#     minimum_spanning_tree: Set[Edge] = set()
#     for i in range(0, g.vertex_num):
#         vertice = i
#         make_set(vertice)
#         # edges = list(graph["edges"])
#         edges = g.edge_list
#         edges.sort(key=lambda edge: edge.weight)
#         # print edges
#         for edge in edges:
#             if find(edge.a) != find(edge.b):
#                 union(edge.a, edge.b)
#             minimum_spanning_tree.add(edge)

#     return minimum_spanning_tree


# # Алгоритм Краскала
# def kruskal(g: Graph) -> Set[Edge]:
#     vertex_list = [
#         set([i + 1]) for i in range(g.vertex_num)
#     ]  # список множеств вершин, принадлежащих дереву
#     result: Set[Edge] = set()  # список ребер, принадлежащих дереву
#     edge_list = g.edge_list
#     print("start sorting")
#     print(edge_list)
#     print("end sorting")

#     for e in edge_list:
#         edge_list.sort(key=lambda e: e.weight)

#         a_location = vertex_search(vertex_list, e.a.number + 1)
#         b_location = vertex_search(vertex_list, e.b.number + 1)

#         if a_location != b_location:
#             result.add(e)
#             vertex_list[a_location].update(vertex_list[b_location])
#             vertex_list.pop(b_location)

#         if len(vertex_list) == 1:
#             break

#     return result

# # Список множеств вершин, принадлежащих дереву
# vertex_list: List[Set[int]] = [set()] * graph.vertex_num
# for i in range(0, graph.vertex_num):
#     vertex_list[i].add(i + 1)

# # Список рёбер, принадлежащих дереву
# result: Set[Edge] = set()
# edge_list: List[Edge] = graph.edge_list

# edge_list = sorted(edge_list, key=cmp_to_key(edge_comparator))

# for edge in edge_list:
#     a_location = vertex_search(vertex_list, edge.a.number)
#     b_location = vertex_search(vertex_list, edge.b.number)

#     if a_location != b_location:
#         print(edge)

#         result.add(edge)
#         vertex_list[a_location] = set().union(
#             vertex_list[a_location], vertex_list[b_location]
#         )
#         # vertex_list[a_location] = vertex_list[b_location] + vertex_list[a_location]
#         del vertex_list[b_location]

#     if len(vertex_list) == 1:
#         break
# return result


# Алгоритм Прима
# def prim(graph: Graph) -> Set[Edge]:
#     result: Set[Edge] = set()

#     # Начальная вершина - 1
#     tree_vertices: Set[int] = {1}

#     # Список инцидентных дереву рёбер
#     edge_list: List[Edge] = graph.list_of_edges_by_vertex(Vertex(1))

#     # Пока все вершины не добавлены в дерево
#     while len(tree_vertices) != graph.vertex_num:
#         # Сортировка списка рёбер
#         edge_list = sorted(edge_list, key=cmp_to_key(edge_comparator))

#         # Поиск первого ребра, не создающего цикла
#         i = 0
#         while edge_list[i].b.number in tree_vertices:
#             i += 1
#         # Добавление полученного ребра
#         result.add(edge_list[i])

#         # Добавление вершины в дерево
#         tree_vertices.add(edge_list[i].b.number)


#         # Добавление всех инцидентных ребер, не ведущих в дерево, в список
#         for edge in graph.list_of_edges_by_vertex(edge_list[i].b):
#             if edge.b.number not in tree_vertices:
#                 edge_list.append(edge)
#         edge_list.pop(0)
#     return result
# def prim(graph: Graph) -> Set[Edge]:
#     sorted_edge_list = sorted(graph.edge_list, key=lambda x: x.weight)
#     connected_vertices = set()  # множество соединенных вершин
#     # словарь множеств изолированных групп вершин
#     isolated_vertices_groups = {}
#     result: Set[Edge] = set()  # множество ребер остова

#     for edge in sorted_edge_list:
#         # r = [edge.weight, edge.a, edge.b]
#         # проверка для исключения циклов в остове
#         if (
#             edge.a not in connected_vertices
#             or edge.b not in connected_vertices
#         ):
#             # если обе вершины не соединены, то
#             if (
#                 edge.a not in connected_vertices
#                 and edge.b not in connected_vertices
#             ):
#                 # формируем в словаре ключ с номерами вершин
#                 isolated_vertices_groups[edge.a] = [
#                     edge.a,
#                     edge.b,
#                 ]
#                 # и связываем их с одним и тем же списком вершин
#                 isolated_vertices_groups[edge.b] = isolated_vertices_groups[
#                     edge.a
#                 ]
#             else:  # иначе
#                 # если в словаре нет первой вершины, то
#                 if not isolated_vertices_groups.get(edge.a):
#                     # добавляем в список первую вершину
#                     isolated_vertices_groups[edge.b].append(edge.a)
#                     # и добавляем ключ с номером первой вершины
#                     isolated_vertices_groups[
#                         edge.a
#                     ] = isolated_vertices_groups[edge.b]
#                 else:
#                     # иначе, все то же самое делаем со второй вершиной
#                     isolated_vertices_groups[edge.a].append(edge.b)
#                     isolated_vertices_groups[
#                         edge.b
#                     ] = isolated_vertices_groups[edge.a]

#             result.add(edge)  # добавляем ребро в остов
#             # добавляем вершины в множество connected_vertices
#             connected_vertices.add(edge.a)
#             connected_vertices.add(edge.b)

#     # проходим по ребрам второй раз и объединяем разрозненные группы вершин
#     for edge in sorted_edge_list:
#         # r = [edge.weight, edge.a, edge.b]
#         # если вершины принадлежат разным группам, то объединяем
#         if edge.b not in isolated_vertices_groups[edge.a]:
#             result.add(edge)  # добавляем ребро в остов
#             gr1 = isolated_vertices_groups[edge.a]
#             # объединем списки двух групп вершин
#             isolated_vertices_groups[edge.a] += isolated_vertices_groups[
#                 edge.b
#             ]
#             isolated_vertices_groups[edge.b] += gr1

#     return result


# def prim(g: Graph) -> Set[Edge]:
#     result: Set[Edge] = set()
#     tree_vertices = {0}  # начальная вершина - 1
#     edge_list = g.list_of_edges_by_vertex(0)  # список инцидентных дереву ребер

#     # пока все вершины не добавлены в дерево
#     while len(tree_vertices) != g.vertex_num:
#         # сортировка списка ребер
#         print("edge sorting")
#         edge_list.sort(key=cmp_to_key(edge_comparator))
#         print(edge_list)
#         print("end sorting")

#         # поиск первого ребра, не создающего цикла
#         i = 0
#         while edge_list[i].b in tree_vertices:
#             i += 1
#         # добавление полученного ребра
#         result.add(edge_list[i])

#         # добавление вершины в дерево
#         tree_vertices.add(edge_list[i].b)

#         # добавление всех инцидентных ребер, не ведущих в дерево, в список
#         for e in g.list_of_edges_by_vertex(edge_list[i].b):
#             if e.b not in tree_vertices:
#                 edge_list.append(e)
#         # удаление добавленного ребра из списка
#         edge_list.pop(i)
#     return result


def prim(g: Graph) -> List[Edge]:
    # Создаем список для хранения выбранных вершин
    selected: List[bool] = [False] * g.vertex_num
    # Инициализируем список для хранения ребер минимального остовного дерева
    mst: List[Edge | None] = [None] * (g.vertex_num - 1)

    # Помечаем первую вершину как выбранную
    selected[0] = True

    # Итерируемся, пока не выберем все вершины
    for _ in range(g.vertex_num - 1):
        min_weight = INF
        u, v = -1, -1

        # Ищем ребро с минимальным весом, где одна вершина уже выбрана, а другая - нет
        for i in range(g.vertex_num):
            if selected[i]:
                for j in range(g.vertex_num):
                    if not selected[j] and g.adj_matrix[i][j]:
                        if g.adj_matrix[i][j] < min_weight:
                            min_weight = g.adj_matrix[i][j]
                            u, v = i, j

        # Добавляем найденное ребро к минимальному остовному дереву
        mst[_] = Edge(u, v, g.adj_matrix[u][v])
        selected[v] = True

    return mst  # type: ignore


# АЛГОРИТМ КРАСКАЛА
def kruskal(g: Graph) -> List[Edge]:
    def find(parent, i):
        # Находим корень вершины i
        while parent[i] != i:
            i = parent[i]
        return i

    def union(parent, rank, x, y):
        # Объединяем два множества по рангу
        x_root = find(parent, x)
        y_root = find(parent, y)

        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        elif rank[x_root] > rank[y_root]:
            parent[y_root] = x_root
        else:
            parent[y_root] = x_root
            rank[x_root] += 1

    # Инициализируем родительский массив и массив рангов
    parent = [i for i in range(g.vertex_num)]
    rank = [0] * g.vertex_num
    # Инициализируем список для хранения ребер минимального остовного дерева
    mst: List[Edge] = []

    # Сортируем ребра графа по возрастанию весов
    edges: List[Edge] = sorted(g.edge_list, key=lambda edge: edge.weight)

    # Проходим по всем ребрам и добавляем их к минимальному остовному дереву,
    # если они не создадут цикл
    for edge in edges:
        # u, v, weight = edge
        u_root = find(parent, edge.a)
        v_root = find(parent, edge.b)

        if u_root != v_root:
            # mst.append((u, v))
            mst.append(edge)
            union(parent, rank, u_root, v_root)

    return mst


# Алгоритм Борувки
def boruvka(g: Graph):
    # Функция для поиска компонента с минимальным весом ребра
    def find_min_edge(adj_matrix: List[List[int]], comp: Set[int]) -> Edge:
        # min_edge = [None, None, float("inf")]
        # min_edge = Edge(-1, -1, INF)
        minimum_edge = Edge(-1, -1, INF)
        for u in comp:
            for v in range(g.vertex_num):
                if (
                    adj_matrix[u][v] != 0
                    and adj_matrix[u][v] < minimum_edge.weight
                    and v not in comp
                ):
                    # min_edge = [u, v, graph[u][v]]
                    minimum_edge = Edge(u, v, adj_matrix[u][v])
        return minimum_edge

    # Функция для объединения компонент
    def union_components(components, comp1, comp2):
        components[comp1] |= components[comp2]
        components.pop(comp2)

    components: List[Set[int]] = [{v} for v in range(g.vertex_num)]
    min_spanning_tree = []

    while len(components) > 1:
        # cheapest = [None, None, float("inf")]
        cheapest = Edge(-1, -1, INF)

        # Находим минимальные ребра для каждой компоненты
        for comp in components:
            min_edge = find_min_edge(g.adj_matrix, comp)
            if min_edge.weight < cheapest.weight:
                cheapest = min_edge

        comp1 = None
        comp2 = None

        # Находим компоненты, к которым принадлежат концы минимального ребра
        for i, comp in enumerate(components):
            if cheapest.a in comp:
                comp1 = i
            if cheapest.b in comp:
                comp2 = i

        # Если компоненты различны, добавляем ребро в минимальное остовное дерево
        if comp1 != comp2:
            min_spanning_tree.append(cheapest)
            union_components(components, comp1, comp2)

    return min_spanning_tree


# Алгоритм Дейкстры
# def dijkstra(start: int, end: int, g: Graph) -> Tuple[int, List[Edge]]:
#     marks = [INF] * g.vertex_num
#     marks[start - 1] = 0

#     visited = [False] * g.vertex_num
#     prev = [-1] * g.vertex_num
#     prev[start - 1] = start

#     reached = False

#     while visited.count(False) != 0:
#         minVertex = -1
#         minMark = INF
#         for i in range(len(marks)):
#             if not visited[i] and marks[i] < minMark:
#                 minMark = marks[i]
#                 minVertex = i + 1

#         for e in g.list_of_edges_by_vertex(minVertex):
#             if not visited[e.b - 1] and marks[e.b - 1] > marks[e.a - 1] + e.weight:
#                 marks[e.b - 1] = marks[e.a - 1] + e.weight
#                 prev[e.b - 1] = e.a
#                 if e.b == end:
#                     reached = True

#         visited[minVertex - 1] = True

#         if marks.count(INF) == visited.count(False):
#             if not reached:
#                 return -1, []
#             else:
#                 break

#     path = []
#     length = 0
#     i = end - 1
#     while prev[i] != i + 1:
#         e = g.get_edge(prev[i], i + 1)
#         length += e.weight
#         path.append(e)
#         i = prev[i] - 1

#     path.reverse()
#     return length, path


# Алгоритм Дейкстры
# def dijkstraa(start: int, end: int, g: Graph) -> Tuple[int, List[Edge]]:
#     INF = 9999999

#     def arg_min(T, S):
#         amin = -1
#         m = INF  # максимальное значение
#         for i, t in enumerate(T):
#             if t < m and i not in S:
#                 m = t
#                 amin = i

#         return amin

#     N = g.vertex_num  # число вершин в графе
#     T = [INF] * N  # последняя строка таблицы

#     v = 0  # стартовая вершина (нумерация с нуля)
#     S = {v}  # просмотренные вершины
#     T[v] = 0  # нулевой вес для стартовой вершины
#     M = [0] * N  # оптимальные связи между вершинами

#     while v != -1:  # цикл, пока не просмотрим все вершины
#         for j, dw in enumerate(
#             g.adj_matrix[v]
#         ):  # перебираем все связанные вершины с вершиной v
#             if j not in S:  # если вершина еще не просмотрена
#                 w = T[v] + dw
#                 if w < T[j]:
#                     T[j] = w
#                     M[j] = v  # связываем вершину j с вершиной v

#         v = arg_min(T, S)  # выбираем следующий узел с наименьшим весом
#         if v >= 0:  # выбрана очередная вершина
#             S.add(v)  # добавляем новую вершину в рассмотрение

#     # print(T, M, sep="\n")

#     # формирование оптимального маршрута:
#     start = 0
#     end = 4
#     P = [end]
#     while end != start:
#         end = M[P[-1]]
#         P.append(end)

#     return (1, P)


def dijkstra(start: int, end: int, g: Graph) -> Tuple[int, List[Edge]]:
    # ИНДЕКСЫ
    start = start - 1
    end = end - 1
    current_path = 0
    paths: List[int] = [INF] * g.vertex_num
    visited: List[bool] = [False] * g.vertex_num
    prev = [-1] * g.vertex_num

    def find_min_vertex():
        min_value = INF + 1
        min_vertex = -1
        for i in range(g.vertex_num):
            if paths[i] < min_value and not visited[i]:
                min_value = paths[i]
                min_vertex = i
        return min_vertex

    paths[start] = 0
    prev[start] = start

    # Ищем само расстояние
    for i in range(g.vertex_num):
        start = find_min_vertex()
        current_path = paths[start]
        for edge in g.list_of_edges_by_vertex(start):
            if current_path + edge.weight < paths[edge.b]:
                paths[edge.b] = current_path + edge.weight
                prev[edge.b] = edge.a
        visited[start] = True

    # Если не дошли
    final_length = paths[end]
    if final_length == INF:
        return (INF, [])

    # Ищем обратный путь
    path: List[Edge] = []
    length = final_length

    prev_vertex = prev[end]
    edge = g.get_edge(prev_vertex, end)
    path.append(Edge(prev_vertex, end, edge.weight))
    length -= edge.weight
    while length > 0:
        end = prev_vertex
        prev_vertex = prev[prev_vertex]
        edge = g.get_edge(prev_vertex, end)
        path.append(Edge(prev_vertex, end, edge.weight))
        length -= edge.weight

    path.reverse()
    return (final_length, path)


# Алгоритм дейкстры, возвращает
def dijkstra_for_distances(start: int, g: Graph) -> List[int]:
    # ИНДЕКСЫ
    start = start - 1
    current_path = 0
    paths: List[int] = [INF] * g.vertex_num
    visited: List[bool] = [False] * g.vertex_num
    prev = [-1] * g.vertex_num

    def find_min_vertex():
        min_value = INF + 1
        min_vertex = -1
        for i in range(g.vertex_num):
            if paths[i] < min_value and not visited[i]:
                min_value = paths[i]
                min_vertex = i
        return min_vertex

    paths[start] = 0
    prev[start] = start

    # Ищем само расстояние
    for i in range(g.vertex_num):
        start = find_min_vertex()
        current_path = paths[start]
        for edge in g.list_of_edges_by_vertex(start):
            if current_path + edge.weight < paths[edge.b]:
                paths[edge.b] = current_path + edge.weight
                prev[edge.b] = edge.a
        visited[start] = True
    return paths


# Алгоритм Беллмана-Форда-Мура
def bellman_ford_moore(start: int, g: Graph) -> List[int]:
    start = start - 1  # индекс начальной вершины
    distances = [INF] * g.vertex_num
    distances[start] = 0

    # Идём по каждой оставшейся вершине
    for i in range(0, g.vertex_num - 1):
        # Идём по каждому ребру
        for edge in g.edge_list:
            if distances[edge.b] > distances[edge.a] + edge.weight:
                distances[edge.b] = distances[edge.a] + edge.weight
    return distances


def find_neg_cycle(g: Graph) -> bool:
    dist = bellman_ford_moore(1, g)

    for edge in g.edge_list:
        if dist[edge.b] > dist[edge.a] + edge.weight:
            return True
    return False


# def levit(start: int, g: Graph):
#     start = start - 1  # ПОЛУЧАЕМ ИНДЕКС НАЧАЛЬНОЙ ВЕРШИНЫ

#     m0: List[int] = []
#     m1: List[int] = []
#     m2: List[int] = []

#     dist: List[int] = [INF] * g.vertex_num

#     for i in range(g.vertex_num):
#         if i == start:
#             m1.append(i)
#         else:
#             m0.append(i)

#     while len(m1) > 0:
#         current_vertex = m1[0]
#         m1 = m1[1::]
#         m2.append(current_vertex)

#         for edge in g.list_of_edges_by_vertex(current_vertex):
#             next_vertex = edge.b
#             if next_vertex in m0:
#                 dist[next_vertex] = dist[current_vertex] + edge.weight
#                 m0 = [element for element in m0 if element != next_vertex]
#                 m1.append(next_vertex)
#             elif next_vertex in m1:
#                 dist[next_vertex] = min(
#                     dist[next_vertex], dist[current_vertex] + edge.weight
#                 )
#             else:
#                 if dist[next_vertex] > dist[current_vertex] + edge.weight:
#                     dist[next_vertex] = dist[current_vertex] + edge.weight
#                     m2 = [element for element in m2 if element != next_vertex]
#                     m1.append(next_vertex)
#     return dist


def levit(start: int, g: Graph):
    start = start - 1
    m0 = deque()
    m1 = deque()
    m2 = deque()

    dist = [INF] * g.vertex_num
    dist[start] = 0

    for i in range(g.vertex_num):
        if i == start:
            m1.append(i)
        else:
            m0.append(i)

    while m1:
        vCur = m1.popleft()
        m2.append(vCur)

        for e in g.list_of_edges_by_vertex(vCur):
            vNext = e.b
            if vNext in m0:
                dist[vNext] = dist[vCur] + e.weight
                m0.remove(vNext)
                m1.append(vNext)
            elif vNext in m1:
                dist[vNext] = min(dist[vNext], dist[vCur] + e.weight)
            else:
                if dist[vNext] > dist[vCur] + e.weight:
                    dist[vNext] = dist[vCur] + e.weight
                    if vNext in m2:
                        m2.remove(vNext)
                        m1.append(vNext)

    return dist


# Алгоритм Джонсона
def johnson(g: Graph) -> List[List[int]]:
    result: List[List[int]] = []
    n = g.vertex_num

    h: List[int] = [INF] * (n + 1)
    h[n] = 0

    edited_edge_list: List[Edge] = deepcopy(g.edge_list)
    for i in range(n):
        edited_edge_list.append(Edge(n, i, 0))

    # Алгоритм Беллмана-Форда из вершины n + 1
    for i in range(n + 1):
        for edge in edited_edge_list:
            h[edge.b] = min(h[edge.b], h[edge.a] + edge.weight)

    # Пересчёт весов исходного графа
    for edge in g.edge_list:
        edge.weight += h[edge.a] - h[edge.b]

    # Алгоритм Дейкстры
    for i in range(n):
        dist = dijkstra_for_distances(i + 1, g)
        true_dist: List[int] = []

        for j in range(n):
            true_dist.append(dist[j] - h[i] + h[j])
        result.append(true_dist)

    return result
