from graph import Graph
from typing import List, Set, Tuple
from edge import Edge
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
        u_root = find(parent, edge.a)
        v_root = find(parent, edge.b)

        if u_root != v_root:
            mst.append(edge)
            union(parent, rank, u_root, v_root)

    return mst


# Алгоритм Борувки
def boruvka(g: Graph):
    # Функция для поиска компонента с минимальным весом ребра
    def find_min_edge(adj_matrix: List[List[int]], comp: Set[int]) -> Edge:
        minimum_edge = Edge(-1, -1, INF)
        for u in comp:
            for v in range(g.vertex_num):
                if (
                    adj_matrix[u][v] != 0
                    and adj_matrix[u][v] < minimum_edge.weight
                    and v not in comp
                ):
                    minimum_edge = Edge(u, v, adj_matrix[u][v])
        return minimum_edge

    # Функция для объединения компонент
    def union_components(components, comp1, comp2):
        components[comp1] |= components[comp2]
        components.pop(comp2)

    components: List[Set[int]] = [{v} for v in range(g.vertex_num)]
    min_spanning_tree = []

    while len(components) > 1:
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


# Алгоритм дейкстры, возвращает расстояние до вершин
def dijkstra_for_distances(start: int, g: Graph) -> List[int]:
    start = start - 1  # Получаем индекс стартовой вершины
    current_path = 0  # Текущая длина пути
    paths: List[int] = [INF] * g.vertex_num  # Расстояния до остальных вершин
    visited: List[bool] = [False] * g.vertex_num  # Посещённость вершин
    prev = [-1] * g.vertex_num  # Предыдущие вершины для каждой вершины

    # Поиск вершины, расстояние до которой минимально
    def find_min_vertex():
        min_value = INF + 1  # Значение для поиска минимума
        min_vertex = -1  # Вершина, расстояние до которой минимально
        # Идём по всем вершинам
        for i in range(g.vertex_num):
            # Если расстояние до вершины меньше минимума и вершина не посещена
            if paths[i] < min_value and not visited[i]:
                # Обновляем нминимальный путь
                min_value = paths[i]
                # Обновляем минимальную вершину
                min_vertex = i
        return min_vertex

    # Расстояние до начальной вершины равно нулю
    paths[start] = 0
    # Предыдущая вершина для стартовой - стартовая
    prev[start] = start

    # Ищем само расстояние
    for i in range(g.vertex_num):
        # Ищем следующую минимальную вершину
        start = find_min_vertex()
        # Текущий путь до вершины
        current_path = paths[start]
        # Идём по всем рёбрам, которые инцидентны текущей вершине
        for edge in g.list_of_edges_by_vertex(start):
            # До каждой вершины обновляем путь
            if current_path + edge.weight < paths[edge.b]:
                paths[edge.b] = current_path + edge.weight
                prev[edge.b] = edge.a
        # Отмечаем текущую вершину посещённой
        visited[start] = True
    return paths


# Алгоритм Беллмана-Форда-Мура
def bellman_ford_moore(start: int, g: Graph) -> List[int]:
    start = start - 1  # индекс начальной вершины
    distances = [INF] * g.vertex_num
    distances[start] = 0

    # Идём по каждой оставшейся вершине
    for i in range(0, g.vertex_num - 1):
        # Идём по каждому ребру графа
        for edge in g.edge_list:
            if distances[edge.b] > distances[edge.a] + edge.weight:
                distances[edge.b] = distances[edge.a] + edge.weight
    return distances


# Ищем отрицательный цикл
def find_neg_cycle(g: Graph) -> bool:
    # Идём алгоритмом Беллма-Форда-Мура
    dist = bellman_ford_moore(1, g)

    # И идём ещё раз
    for edge in g.edge_list:
        if dist[edge.b] > dist[edge.a] + edge.weight:
            return True
    return False


def levit(start: int, g: Graph):
    # Индекс начальной вершины
    start = start - 1
    m0 = deque()  # Вершины, расстояние до которых уже вычислено
    m1 = deque()  # Вершины, расстояние до которых вычисляется
    m2 = deque()  # Вершины, для которых не вычислено

    # Массив расстояний до вершины
    dist = [INF] * g.vertex_num
    # Расстояние до начальной вершины равно нулю
    dist[start] = 0

    # Идём по каждой вершине
    for i in range(g.vertex_num):
        if i == start:
            m1.append(i)
        else:
            m0.append(i)

    # Пока есть вершины, расстояние до которых вычисляется
    while m1:
        # Текущая веришна - первая в очереди тех, расстояние для которых вычисляется
        current_vertex = m1.popleft()
        m2.append(current_vertex)

        # Идём по всем рёбрам, инцидентным данной вершине
        for e in g.list_of_edges_by_vertex(current_vertex):
            # Соседняя вершина
            next_vertex = e.b
            # Если следующая вершина есть в очереди вершин, расстояние до которых вычислено
            if next_vertex in m0:
                dist[next_vertex] = dist[current_vertex] + e.weight
                m0.remove(next_vertex)
                m1.append(next_vertex)
            # Если вершина есть в очереди, расстояние до которых вычисляется
            elif next_vertex in m1:
                # Обновляем расстояние до вершины
                dist[next_vertex] = min(
                    dist[next_vertex], dist[current_vertex] + e.weight
                )
            # Если расстояние до вершины вообще не вычислено
            else:
                if dist[next_vertex] > dist[current_vertex] + e.weight:
                    dist[next_vertex] = dist[current_vertex] + e.weight
                    if next_vertex in m2:
                        m2.remove(next_vertex)
                        m1.append(next_vertex)

    return dist


# Алгоритм Джонсона
def johnson(g: Graph) -> List[List[int]]:
    # Матрица результата
    result: List[List[int]] = []
    # Количество вершин в графе
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
