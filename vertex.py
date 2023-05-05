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
