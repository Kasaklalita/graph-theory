from enum import Enum
from graph import Graph
from vertex import Vertex
from typing import List


class InputType(Enum):
    ADJ_MATRIX = 1
    ADJ_LIST = 2
    EDGE_LIST = 3


def split_row(line: str):
    ints = line.split(" ")
    return [int(entity) for entity in ints]


def print_help_info():
    pass


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
