from graph import Graph
from typing import List, Set
from vertex import Vertex
from edge import Edge


# Поиск в ширину
def BFS(g: Graph, start: Vertex, visited: List[bool], container: List[int]):
    queue: List[Vertex] = []
    queue.append(start)

    visited[start.number - 1] = True

    while len(queue) > 0:
        y = queue.pop()
        container.append(y.number)

        for edge in g.list_of_edges_by_vertex(y):
            if not visited[edge.b.number - 1]:
                queue.append(edge.b)
                visited[edge.b.number - 1] = True


# Подсчёт компонент связности
def connectivity_components_count(graph: Graph):
    # Поиск в ширину для подсчёта компонент связности
    def BFS(g: Graph, start: Vertex, visited: List[int]):
        queue: List[int] = []  # Очередь
        queue.append(start.number)

        # Добавление стартовой вершины в список посещённх
        visited.append(start.number)

        # Пока очередь не пуста
        while len(queue) != 0:
            # Получение крайнего элемента в очереди
            y = queue.pop()

            # Идём по всем рёбрам, содержащим вершину y
            for edge in g.list_of_edges_by_vertex(Vertex(y)):
                # Если другая вершина не посещена, то добавляем её в очередь
                if edge.b.number not in visited:
                    queue.append(edge.b.number)
                    visited.append(edge.b.number)

    visited: List[int] = []  # Список посещённых вершин
    result = 0  # Количество компонент связности
    for i in range(0, graph.vertex_num):
        if (i + 1) not in visited:
            BFS(graph, Vertex(i + 1), visited)
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

    return [i + 1 for i, is_art in enumerate(is_articulation) if is_art]


# Поиск мостов в графе
def find_bridges(graph: Graph, joints: Set[int]) -> Set[Edge]:
    result: Set[Edge] = set()  # Результат

    # Идём по всем шарнирам
    for joint in joints:
        current_vertex = Vertex(joint)
        # Список всех рёбер, исходящих из вершины
        edge_list = graph.list_of_edges_by_vertex(current_vertex)
        # Подсчёт компонент в исходном графе
        prev_connectivity_components_count = connectivity_components_count(
            graph
        )

        # Идём по всем рёбрам вершины
        for edge in edge_list:
            # Отзеркаливаем ребро, чтобы удалить и его
            mirrored_edge = Edge(edge.b.number, edge.a.number, edge.weight)
            # Удаление обоих рёбер
            graph.delete_edge(edge)
            graph.delete_edge(mirrored_edge)
            # Считаем компоненты связности
            current_connectivity_components_number = (
                connectivity_components_count(graph)
            )
            # Если число компонент увеличилось и это ребро ещё не смотрели
            if (
                current_connectivity_components_number
                > prev_connectivity_components_count
            ) and mirrored_edge not in result:
                result.add(edge)

            # Восстанавливаем рёбра
            graph.add_edge(edge)
            graph.add_edge(mirrored_edge)

    return result


def find_bridgess(graph: Graph):
    adj_matrix = graph.adj_matrix
    n = len(adj_matrix)
    bridges: List[Edge] = []
    visited = [False] * n
    entry_time = [0] * n
    exit_time = [0] * n
    time = 0

    def dfs(node, parent):
        nonlocal time
        visited[node] = True
        entry_time[node] = time
        exit_time[node] = time
        time += 1
        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and neighbor != parent:
                if not visited[neighbor]:
                    dfs(neighbor, node)
                    exit_time[node] = min(exit_time[node], exit_time[neighbor])
                    if entry_time[node] < exit_time[neighbor]:
                        bridges.append(
                            Edge(
                                node + 1,
                                neighbor + 1,
                                graph.adj_matrix[node][neighbor],
                            )
                        )
                else:
                    exit_time[node] = min(
                        exit_time[node], entry_time[neighbor]
                    )

    for node in range(n):
        if not visited[node]:
            dfs(node, None)

    return bridges
