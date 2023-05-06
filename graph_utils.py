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
def find_joints(graph: Graph) -> Set[int]:
    # Результат
    result: Set[int] = set()

    # Подсчёт компонент связности в исходном графе
    prev_connectivity_components_count = connectivity_components_count(graph)

    # Идём по всем вершинам
    for i in range(1, graph.vertex_num + 1):
        # Список удалённых рёбер
        deleted_edges: List[Edge] = []

        # Идём по всем рёбрам
        for edge in graph.edge_list:
            # Если ребро содержит очередную вершину
            if edge.contains(Vertex(i)):
                # Удаление ребра из списка
                deleted_edges.append(edge)
                graph.delete_edge(edge)

        # Подсчёт компонент после удаления вершины. -1 потому что вершина тоже компонента
        current_connectivity_components_count = connectivity_components_count(graph) - 1

        # Если число увеличилось, то вершина - шарнир
        if current_connectivity_components_count > prev_connectivity_components_count:
            result.add(i)

        # Восстановление рёбер
        for edge in deleted_edges:
            graph.add_edge(edge)
    return result


# Поиск мостов в графе
def find_bridges(graph: Graph, joints: Set[int]) -> Set[Edge]:
    result: Set[Edge] = set()  # Результат

    # Идём по всем шарнирам
    for joint in joints:
        current_vertex = Vertex(joint)
        # Список всех рёбер, исходящих из вершины
        edge_list = graph.list_of_edges_by_vertex(current_vertex)
        # Подсчёт компонент в исходном графе
        prev_connectivity_components_count = connectivity_components_count(graph)

        # Идём по всем рёбрам вершины
        for edge in edge_list:
            # Отзеркаливаем ребро, чтобы удалить и его
            mirrored_edge = Edge(edge.b.number, edge.a.number, edge.weight)
            # Удаление обоих рёбер
            graph.delete_edge(edge)
            graph.delete_edge(mirrored_edge)
            # Считаем компоненты связности
            current_connectivity_components_number = connectivity_components_count(
                graph
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
