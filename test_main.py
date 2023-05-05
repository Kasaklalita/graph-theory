from graph import Vertex, Edge, split_row


def test_vertices():
    v1 = Vertex(1)
    v2 = Vertex(2)
    assert v1.number == 1
    assert v2.number == 2
    assert v1 < v2


def test_edges():
    edge = Edge(2, 5, 1)
    assert edge.a.number == 2
    assert edge.b.number == 5
    assert edge.weight == 1
    assert edge.print_edge() == '2 5 1'


def test_split():
    assert split_row('1 2 3') == [1, 2, 3]
