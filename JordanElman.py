import random
import numpy as np
import math

def factorial(n):
    if n < 0:
        return None

    elif n == 0 or n == 1:
        return 1

    return n * factorial(n - 1)

def fib(n):
    if n == 0:
        return 0

    elif n == 1:
        return 1

    return fib(n - 1) + fib(n - 2)

def cos(n):
    return round(math.cos(n * math.pi / 2))

def pow_fun(n):
    return 1.0625 ** n

def fact_fun(n):
    return math.log(factorial(n)) / math.log(10)
               

if __name__ == '__main__':
    k = 10

    sequences = [
        [fib(i) for i in range(k)],
        [fact_fun(i) for i in range(k)],
        [cos(i) for i in range(k)],
        [pow_fun(i) for i in range(k)]
    ]

    sequence_functions = [fib, fact_fun, cos, pow_fun]

    sequence_names = ['Fibonacci', 'log_2 n!', 'Sin(i * pi / 2)', '1.125^n']

    q = k - 1
    p = 1
    L = q - p 
    learning_rate = 0.0007
    N = 100000

    print("Sequences:")
    for index, i in enumerate(sequences):
        print(f"{index + 1})", i)

    choice = int(input("Select sequence: "))

    if (choice < 1):
        choice = 1

    elif (choice > len(sequences)):
        choice = len(sequences)
    
    current_sequence = sequences[choice - 1]
    
    train_row, train_col = L, p + 1 

    train_matrix = np.empty((train_row, train_col))
    train_etalons = np.empty((train_row, train_col))

    for i in range(train_row):
        for j in range(train_col):
            train_matrix[i, j] = current_sequence[i + j]
            train_etalons[i, j] = current_sequence[i + j + 1]

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    ilayer_size = p + 1
    hlayer_size = int(ilayer_size / 2)
    olayer_size = ilayer_size
         
    w_size = p + 1

    m_tb_first = 0
    v_tb_first = 0

    m_tb_second = 0
    v_tb_second = 0

    m_t_first = 0
    v_t_first = 0

    m_t_second = 0
    v_t_second = 0

    m_t_third = 0
    v_t_third = 0

    m_t_fourth = 0
    v_t_fourth = 0

    prevDb1 = 0
    prevDb2 = 0

    prevDW1 = 0
    prevDW2 = 0
    prevDW3 = 0
    prevDW4 = 0

    GW1 = 0
    GW2 = 0
    GW3 = 0
    GW4 = 0

    previous_hidden = np.empty((1, hlayer_size))
    previous_output = np.empty((1, olayer_size))

    b_first = np.random.uniform(size=(1, hlayer_size), low=-1, high=1)
    bW_first = np.ones((hlayer_size, hlayer_size))

    b_second = np.random.uniform(size=(1, olayer_size), low=-1, high=1)
    bW_second = np.ones((olayer_size, olayer_size))

    W_first = np.random.uniform(size=(ilayer_size, hlayer_size), low=-1, high=1)
    W_second = np.random.uniform(size=(hlayer_size, olayer_size), low=-1, high=1)
    W_third = np.random.uniform(size=(ilayer_size, hlayer_size), low=-1, high=1)
    W_fourth = np.random.uniform(size=(hlayer_size, hlayer_size), low=-1, high=1)

    # =====================================================

    error = 10000000
    iteration_number = 0
    while iteration_number < N:
        error = 0

        for input_sequence in range(len(train_matrix)):
            data = np.empty((1, ilayer_size))
            data[0][...] = train_matrix[input_sequence]

            hidden = data @ W_first + previous_output @ W_third + previous_hidden @ W_fourth + b_first @ bW_first
            out = hidden @ W_second + b_second @ bW_second

            delta = out - train_etalons[input_sequence]

            b_first_delta = delta @ W_second.T
            b_second_delta = delta

            W_first_delta =  data.T @ delta @ W_second.T
            W_second_delta = hidden.T @ delta
            W_third_delta = previous_output.T @ delta @ W_second.T
            W_fourth_delta = previous_hidden.T @ delta @ W_second.T

            previous_output = out
            previous_hidden = hidden

            m_tb_first = beta1 * m_tb_first + (1 - beta1) * b_first_delta
            v_tb_first = beta2 * v_tb_first + (1 - beta2) * b_first_delta ** 2

            m_tb_second = beta1 * m_tb_second + (1 - beta1) * b_second_delta
            v_tb_second = beta2 * v_tb_second + (1 - beta2) * b_second_delta ** 2

            m_t_first = beta1 * m_t_first + (1 - beta1) * W_first_delta
            v_t_first = beta2 * v_t_first + (1 - beta2) * W_first_delta ** 2

            m_t_second = beta1 * m_t_second + (1 - beta1) * W_second_delta
            v_t_second = beta2 * v_t_second + (1 - beta2) * W_second_delta ** 2

            m_t_third = beta1 * m_t_third + (1 - beta1) * W_third_delta
            v_t_third = beta2 * v_t_third + (1 - beta2) * W_third_delta ** 2

            m_t_fourth = beta1 * m_t_fourth + (1 - beta1) * W_fourth_delta
            v_t_fourth = beta2 * v_t_fourth + (1 - beta2) * W_fourth_delta ** 2


            prevDb1 = - learning_rate / (np.sqrt(v_tb_first) + eps) * m_tb_first
            prevDb2 = - learning_rate / (np.sqrt(v_tb_second) + eps) * m_tb_second

            prevDW1 = - learning_rate / (np.sqrt(v_t_first) + eps) * m_t_first
            prevDW2 = - learning_rate / (np.sqrt(v_t_second) + eps) * m_t_second
            prevDW3 = - learning_rate / (np.sqrt(v_t_third) + eps) * m_t_third
            prevDW4 = - learning_rate / (np.sqrt(v_t_fourth) + eps) * m_t_fourth

            b_first += prevDb1
            b_second += prevDb2

            W_first += prevDW1
            W_second += prevDW2
            W_third += prevDW3    
            W_fourth += prevDW4

            data[0][...] = train_matrix[input_sequence]

            hidden = data @ W_first + previous_output @ W_third + previous_hidden @ W_fourth + b_first @ bW_first
            out = hidden @ W_second + b_second @ bW_second

            delta = out - train_etalons[input_sequence]
            error += delta @ delta.T
            
        iteration_number += 1
        print("Epoch #", iteration_number, "Loss: ", error)
               

    # =====================================================

    while True:
        start = random.randint(k, k + 20)

        sequence = [sequence_functions[choice - 1](i) for i in range(start, start + p + 1)]

        print("Sequence name -", sequence_names[choice - 1])
        print("Sequence for test", sequence)

        data = np.empty((1, ilayer_size))
        data[0][...] = sequence

        hidden = data @ W_first + previous_output @ W_third + previous_hidden @ W_fourth + b_first @ bW_first
        out = hidden @ W_second + b_second @ bW_second

        previous_hidden = hidden
        previous_output = out

        print("Network output: ", out)
        print("What you should see: ", [sequence_functions[choice - 1](i) for i in range(start + 1, start + p + 2)])

        answer = input("Do you want to repeat test?[Y/N]")

        if (answer == 'N' or answer == 'n'):
            break