from graph import Graph
from utils import InputType, print_help_info
import sys
from graph_utils import find_joints, find_bridges


# DONE
def main():
    inputType: InputType = InputType.EDGE_LIST
    inputPath: str
    outputPath: str = ""
    key: str

    inputKeyExists = False
    multipleInputkeys = False
    helpKeyExists = False
    outputKeyExists = False

    args = sys.argv[1:]
    for i in range(0, len(args)):
        key = args[i]
        # Если несколько ключей, то ошибка
        if (key == "-e" or key == "-m" or key == "-l") and inputKeyExists:
            multipleInputkeys = True
        # Первый ключ для ввода найден
        elif key == "-e" or key == "-m" or key == "-l":
            inputKeyExists = True

        # Если ключ помощи, то все остальные не нужны
        if key == "-h":
            helpKeyExists = True
            break

        # Ключ для вывода в файл
        if key == "-o":
            outputKeyExists = True
            outputPath = args[i + 1]

    # Вывод помощи
    if helpKeyExists:
        print_help_info()
        return 0

    # Проверки на ошибки
    if not inputKeyExists:
        print("Ошибка. Нет ключа для ввода данных")
    if multipleInputkeys:
        print("Ошибка. Несколько ключей для ввода данных")

    key = args[0]
    match key:
        case "-e":
            inputType = InputType.EDGE_LIST
        case "-m":
            inputType = InputType.ADJ_MATRIX
        case "-l":
            inputType = InputType.ADJ_LIST

    inputPath = args[1]

    # Создание графа
    g = Graph(inputPath, inputType)

    # Ищем шарниры и мосты
    joints = find_joints(g)
    bridges = find_bridges(g)

    if outputKeyExists:
        fout = open(outputPath, "w")
        fout.write(
            f"Мосты: {[bridge.info_as_bridge() for bridge in bridges]}\n"
        )
        fout.write(f"Шарниры: {[joint for joint in joints]}\n")
    else:
        print(f"Мосты: {[bridge.info_as_bridge() for bridge in bridges]}")
        print(f"Шарниры: {[joint + 1 for joint in joints]}")


if __name__ == "__main__":
    main()
