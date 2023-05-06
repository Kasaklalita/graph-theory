from vertex import Vertex


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

    def info_as_bridge(self):
        return f"({self.a}, {self.b})"

    def __str__(self) -> str:
        return f"{self.a.number} {self.b.number} {self.weight}"

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def __hash__(self):
        return hash((self.a, self.b, self.weight))
