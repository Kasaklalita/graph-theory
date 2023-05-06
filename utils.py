from enum import Enum
from typing import List


class InputType(Enum):
    ADJ_MATRIX = 1
    ADJ_LIST = 2
    EDGE_LIST = 3


def split_row(line: str) -> List[int]:
    line = line.strip()
    if len(line) == 0:
        return []
    ints = line.split(" ")
    return [int(entity, base=10) for entity in ints]


def print_help_info():
    print("Полиенко Вениамин Сергеевич, группа М3О-210Б-21")
    print("Список ключей:")
    print("\t-e <путь файла> - граф представлен в виде списка рёбер")
    print("\t-m <путь файла> - граф представлен в виде матрицы смежности")
    print("\t-l <путь файла> - граф представлен в виде списка смежности")
    print(
        "\t-o <путь файла> - выходные данные сохраняются в файл, расположенный по указанному пути"
    )
    print("\t-h - справка\n")
