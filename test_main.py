from graph import Edge, split_row


def test_edges():
    edge = Edge(2, 5, 1)
    assert edge.a == 2
    assert edge.b == 5
    assert edge.weight == 1
    assert edge.print_edge() == "2 5 1"


def test_split():
    assert split_row("1 2 3") == [1, 2, 3]
