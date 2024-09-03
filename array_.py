import torch


class array:
    def __init__(self, args):
        if isinstance(args, list):
            # Если первый элемент списка — это список, значит это двумерный массив (матрица)
            if isinstance(args[0], list):
                self.values = [list(row) for row in args]
            else:  # Если элементы списка не списки, значит это одномерный массив (вектор)
                self.values = [args,]
        else:  # Если аргументы — скаляр
            self.values = [args]

    def __getitem__(self, index):
        return array(self.values[index]) if isinstance(self.values[index], list) else self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = list(value)

    def __len__(self):
        return len(self.values)
    
    def __repr__(self):
        return str(self.values)
    
    def __str__(self):
        return str(self.values)

    def shape(self):
        """Возвращает форму матрицы в виде кортежа (количество строк, количество столбцов)."""
        return (len(self.values), len(self.values[0]) if self.values else 0)
    
    def __add__(self, other):
        """Перегрузка оператора + для сложения двух матриц одинакового размера."""
        if isinstance(other, array):
            if self.shape() != other.shape():
                raise ValueError("Matrices must have the same dimensions to be added.")
            result = [
                [self.values[i][j] + other.values[i][j] for j in range(len(self.values[0]))]
                for i in range(len(self.values))
            ]
            return array(result)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'array' and '{}'".format(type(other).__name__))
    
    def __matmul__(self, other):
        """Перегрузка оператора @ для умножения матриц и вектора на матрицу."""
        if isinstance(other, array):
            if len(other.values) > 0 and isinstance(other.values[0], list) and len(other.values[0]) == 1:
                return self.vector_multiply(other)
            else:
                return self.matrix_multiply(other)
        else:
            raise TypeError("Unsupported operand type(s) for @: 'array' and '{}'".format(type(other).__name__))
    
    def vector_multiply(self, vector):
        """Умножение матрицы на вектор."""
        result = []
        for row in self.values:
            sum_product = sum(row[i] * vector.values[i][0] for i in range(len(vector.values)))
            result.append([sum_product])
        return array(result)

    def matrix_multiply(self, other):
        """Умножение матрицы на матрицу."""
        if len(self.values[0]) != len(other.values):
            raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")
        
        result = []
        for i in range(len(self.values)):
            result_row = []
            for j in range(len(other.values[0])):
                sum_product = sum(self.values[i][k] * other.values[k][j] for k in range(len(other.values)))
                result_row.append(sum_product)
            result.append(result_row)
        return array(result)
    
    def transpose(self):
        """Транспонирование матрицы."""
        transposed = []
        for i in range(len(self.values[0])):
            transposed.append([self.values[j][i] for j in range(len(self.values))])
        return array(transposed)
    
    def inverse(self):
        """Нахождение обратной матрицы с использованием метода Гаусса-Жордана."""
        n = len(self.values)
        identity = [[float(i == j) for i in range(n)] for j in range(n)]
        mat = [row[:] for row in self.values]

        for i in range(n):
            diag_element = mat[i][i]
            if diag_element == 0:
                raise ValueError("Matrix is not invertible.")
            for j in range(n):
                mat[i][j] /= diag_element
                identity[i][j] /= diag_element
            
            for k in range(n):
                if k != i:
                    factor = mat[k][i]
                    for j in range(n):
                        mat[k][j] -= factor * mat[i][j]
                        identity[k][j] -= factor * identity[i][j]

        return array(identity)
    
    @staticmethod
    def eye(size):
        """Создание единичной матрицы размера size x size."""
        return array([[1 if i == j else 0 for j in range(size)] for i in range(size)])
    
    @staticmethod
    def zeros(size):
        """Создание единичной матрицы размера size x size."""
        return array([[0 for j in range(size)] for i in range(size)])
    
    def __mul__(self, other):
        """Перегрузка оператора * для умножения матрицы на скаляр."""
        if isinstance(other, (int, float)):
            result = [
                [self.values[i][j] * other for j in range(len(self.values[0]))]
                for i in range(len(self.values))
            ]
            return array(result)
        elif isinstance(other, array):
            if self.shape() != other.shape():
                raise ValueError("Matrices must have the same dimensions for elementwise multiplication.")
            result = [
                [self.values[i][j] * other.values[i][j] for j in range(len(self.values[0]))]
                for i in range(len(self.values))
            ]
            return array(result)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'array' and '{}'".format(type(other).__name__))



def tensor_to_array(tensor):
    # Преобразуем тензор в список (рекурсивно)
    def tensor_to_list(t):
        if t.dim() == 0:  # Если скаляр
            return t.item()
        elif t.dim() == 1:  # Если одномерный тензор
            return t.tolist()  # Конвертация одномерного тензора в список
        else:  # Если многомерный тензор
            return [tensor_to_list(sub_tensor) for sub_tensor in t]
    
    tensor_list = tensor_to_list(tensor)
    return array(tensor_list)