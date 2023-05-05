from graph import Vertex, Edge, Graph, InputType


def main():
    gr = Graph('adj_matrix.txt', InputType.ADJ_MATRIX)
    print(gr.directed, gr.edge_list.__len__(), gr.adj_matrix, gr.vertex_num)
    gr.print_adjacency_matrix()

if __name__ == "__main__":
    main()
