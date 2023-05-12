from graph import Graph
from typing import List, Set, Tuple
from edge import Edge
from functools import cmp_to_key
from utils import INF


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
        child_count = 0
        visited[u] = True
        disc_time[u] = low[u] = time
        time += 1

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
                    exit_time[node] = min(
                        exit_time[node], entry_time[neighbor]
                    )

    for node in range(n):
        # Начинаем обход с каждой вершины
        if not visited[node]:
            dfs(node, None)

    return bridges


# Компаратор для сортировки ребер по весу
def edge_comparator(a: Edge, b: Edge) -> bool:
    return (
        a.weight < b.weight
        or (a.weight == b.weight and a.a < b.a)
        or (a.weight == b.weight and a.a == b.a and a.b < b.b)
    )


# Поиск вершины в списке множеств. Возвращает позицию множества с найденной вершиной
def vertex_search(vertex_list: List[Set[int]], number: int) -> int:
    for i in range(0, len(vertex_list)):
        if number in vertex_list[i]:
            return i
    return -1


def kruskal(g: Graph) -> Set[Edge]:
    parent = dict()
    rank = dict()

    def make_set(vertice: int):
        parent[vertice] = vertice
        rank[vertice] = 0

    def find(vertice: int):
        if parent[vertice] != vertice:
            parent[vertice] = find(parent[vertice])
        return parent[vertice]

    def union(vertice1: int, vertice2: int):
        root1 = find(vertice1)
        root2 = find(vertice2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
        else:
            parent[root1] = root2
        if rank[root1] == rank[root2]:
            rank[root2] += 1

    # for vertice in graph["vertices"]:
    minimum_spanning_tree: Set[Edge] = set()
    for i in range(0, g.vertex_num):
        vertice = i
        make_set(vertice)
        # edges = list(graph["edges"])
        edges = g.edge_list
        edges.sort(key=lambda edge: edge.weight)
        # print edges
        for edge in edges:
            if find(edge.a) != find(edge.b):
                union(edge.a, edge.b)
            minimum_spanning_tree.add(edge)

    return minimum_spanning_tree


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


def prim(g: Graph) -> Set[Edge]:
    result: Set[Edge] = set()
    tree_vertices = {0}  # начальная вершина - 1
    edge_list = g.list_of_edges_by_vertex(0)  # список инцидентных дереву ребер

    # пока все вершины не добавлены в дерево
    while len(tree_vertices) != g.vertex_num:
        # сортировка списка ребер
        print("edge sorting")
        edge_list.sort(key=cmp_to_key(edge_comparator))
        print(edge_list)
        print("end sorting")

        # поиск первого ребра, не создающего цикла
        i = 0
        while edge_list[i].b in tree_vertices:
            i += 1
        # добавление полученного ребра
        result.add(edge_list[i])

        # добавление вершины в дерево
        tree_vertices.add(edge_list[i].b)

        # добавление всех инцидентных ребер, не ведущих в дерево, в список
        for e in g.list_of_edges_by_vertex(edge_list[i].b):
            if e.b not in tree_vertices:
                edge_list.append(e)
        # удаление добавленного ребра из списка
        edge_list.pop(i)
    return result


# Алгоритм Дейкстры
def dijkstra(start: int, end: int, g: Graph) -> Tuple[int, List[Edge]]:
    marks = [INF] * g.vertex_num
    marks[start - 1] = 0

    visited = [False] * g.vertex_num
    prev = [-1] * g.vertex_num
    prev[start - 1] = start

    reached = False

    while visited.count(False) != 0:
        minVertex = -1
        minMark = INF
        for i in range(len(marks)):
            if not visited[i] and marks[i] < minMark:
                minMark = marks[i]
                minVertex = i + 1

        for e in g.list_of_edges_by_vertex(minVertex):
            if (
                not visited[e.b - 1]
                and marks[e.b - 1] > marks[e.a - 1] + e.weight
            ):
                marks[e.b - 1] = marks[e.a - 1] + e.weight
                prev[e.b - 1] = e.a
                if e.b == end:
                    reached = True

        visited[minVertex - 1] = True

        if marks.count(INF) == visited.count(False):
            if not reached:
                return -1, []
            else:
                break

    path = []
    length = 0
    i = end - 1
    while prev[i] != i + 1:
        e = g.get_edge(prev[i], i + 1)
        length += e.weight
        path.append(e)
        i = prev[i] - 1

    path.reverse()
    return length, path
