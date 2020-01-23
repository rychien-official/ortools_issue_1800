# Test Google OR-Tools w/ CPLEX Backend
# Ryan Chien
# rvc5634@psu.edu
# 1/23/2020

# Imports
import sys
sys.path.append("C:/gitrepo/or-tools/")
from ortools.linear_solver import pywraplp
import numpy as np
dir(pywraplp.Solver)

# Define ILP sudoku solver using CPLEX
def solve_board_CPLEX(initial_board, max_solve_time=120000):
    # libraries
    from ortools.linear_solver import pywraplp
    import numpy as np
    from datetime import datetime

    # Start timer
    start_time = datetime.now()

    # Create solver
    print("(1) Initializing optimization model...")
    solver = pywraplp.Solver(
        "Sudoku",
        pywraplp.Solver.CPLEX_MIXED_INTEGER_PROGRAMMING)
    objective = solver.Objective()
    objective.SetMinimization

    # Get needed range
    board_len = initial_board.__len__()

    # Create objective variable array
    print("(2) Creating objective variables...")
    objective_vars = np.array([
        [solver.IntVar(1, board_len, 'x' + str(j) + str(i)) for i in range(0, board_len)]
        for j in range(0, board_len)])

    # Set initial board constraints
    print("(3) Setting constraints...")
    constraint_initial = []
    for i in range(0, board_len):
        for j in range(0, board_len):
            if initial_board[i][j] != 0:
                constraint_initial.append(
                    solver.Add(
                        objective_vars[i][j] == initial_board[i][j],
                        'init_x'+str(i)+str(j)+'=='+str(initial_board[i][j])))
            else:
                constraint_initial.append(
                    'init_x'+str(i)+str(j)+'==NONE'
                )

    # Set constraint: sum of rows must equal 45 (Not necessary or sufficient, but could help bound.)
    constraint_rowsum = [[]] * board_len
    for i in range(0, board_len):
        constraint_rowsum[i] = solver.Add(
            sum(objective_vars[i]) == sum(range(board_len+1)))

    # Set constraint: sum of columns must equal 45
    constraint_colsum = [[]] * board_len
    for j in range(0, board_len):
        constraint_colsum[j] = solver.Add(
            sum(objective_vars[:, j]) == sum(range(board_len+1)))

    # Set constraint: nonet sums
    # constraint_nonet = []
    if board_len == 9:
        solver.Add(
            sum(sum(objective_vars[0:3, 0:3])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[0:3, 3:6])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[0:3, 6:9])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[3:6, 0:3])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[3:6, 3:6])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[3:6, 6:9])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[6:9, 0:3])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[6:9, 3:6])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[6:9, 6:9])) == sum(range(0, board_len + 1)))


    # Set constraint: t equals difference of objective value pairs row-wise (e.g. t0102 - x01 + x02 = 0)
    t_rows = []
    constraint_t_rows = []
    for i in range(0, board_len):
        t_i = []
        constraint_t_i = []
        for k in range(0, board_len-1):
            t_j = []
            constraint_t_j = []
            for j in range(0+k, board_len-1):
                # one t for each unique objective variable x pair
                t_j.append(
                    solver.IntVar(
                        -10, 10, 't'+str(i)+str(k)+str(i)+str(j+1)))     # range of -10 to 10
                constraint_t_j.append(
                    solver.Add(     # e.g. t0001=x00-x01 ... t0002=x00-x02
                        t_j[j-k]
                        - objective_vars[i][k]
                        + objective_vars[i][j+1]
                        == 0,
                        't'+str(i)+str(k)+str(i)+str(j+1)))
                #print("Constraint: " + constraint_var_t_j[j-k].name() + ' - ' + objective_vars[i][k].name()
                #        + ' + ' + objective_vars[i][j+1].name() + ' = ' + ' 0 ')
            t_i.append(t_j)
            constraint_t_i.append(constraint_t_j)
        t_rows.append(t_i)
        constraint_t_rows.append(constraint_t_i)

    # Create constraint variables p, n, z, and y
    p_rows = []
    n_rows = []
    z_rows = []
    y_rows = []
    for row in t_rows:
        p_i = []
        n_i = []
        z_i = []
        y_i = []
        for column in row:
            p_j = []
            n_j = []
            z_j = []
            y_j = []
            for variable_t in column:
                p_j.append(
                    solver.IntVar(
                        0, 10, 'p'+variable_t.name()[1:5]))
                n_j.append(
                    solver.IntVar(
                        0, 10, 'n'+variable_t.name()[1:5]))
                z_j.append(
                    solver.IntVar(
                        1, 10, 'z'+variable_t.name()[1:5])) # Note that z must be greater than or equal one
                y_j.append(
                    solver.BoolVar(
                        'y'+variable_t.name()[1:5]))
            p_i.append(p_j)
            n_i.append(n_j)
            z_i.append(z_j)
            y_i.append(y_j)
        p_rows.append(p_i)
        n_rows.append(n_i)
        z_rows.append(z_i)
        y_rows.append(y_i)

    # Set constraints: z equal to the absolute value of objective variable pair differences (Big M formulation)
    constraint_tpn_rows = []     # t-p+n=0
    constraint_zpn_rows = []     # z-p-n=0     e.g. z is the absolute value of objective variable pair differences
    constraint_yp_rows = []      # p-10*y<=0   Big-M formulation, where M=10, to enforce either y=0 or p=0
    constraint_yn_rows = []      # n+10*y<=10  Big-M formulation, where M=10, to enforce either y=0 or p=0
    for i in range(0, t_rows.__len__()):
        constraint_tpn_i = []
        constraint_zpn_i = []
        constraint_yp_i = []
        constraint_yn_i = []
        for j in range(0, t_rows[i].__len__()):
            constraint_tpn_j = []
            constraint_zpn_j = []
            constraint_yp_j = []
            constraint_yn_j = []
            for k in range(0, t_rows[i][j].__len__()):
                constraint_tpn_j.append(
                    solver.Add(     # t-p+n=0
                        t_rows[i][j][k]
                        - p_rows[i][j][k]
                        + n_rows[i][j][k]
                        == 0,
                        'tpn'+t_rows[i][j][k].name()[1:5]))
                constraint_zpn_j.append(
                    solver.Add(     # z-p-n=0
                        z_rows[i][j][k]
                        - p_rows[i][j][k]
                        - n_rows[i][j][k]
                        == 0,
                        'zpn'+z_rows[i][j][k].name()[1:5]))
                constraint_yp_j.append(     # p-10*y<=0
                    solver.Add(
                        p_rows[i][j][k]
                        - 10*y_rows[i][j][k]
                        <= 0,
                        'yp'+p_rows[i][j][k].name()[1:5]))
                constraint_yn_j.append(     # n+10*y<=10
                    solver.Add(
                        n_rows[i][j][k]
                        + 10*y_rows[i][j][k]
                        <= 10,
                        'yn' + n_rows[i][j][k].name()[1:5]))
            constraint_tpn_i.append(constraint_tpn_j)
            constraint_zpn_i.append(constraint_zpn_j)
            constraint_yp_i.append(constraint_yp_j)
            constraint_yn_i.append(constraint_yn_j)
        constraint_tpn_rows.append(constraint_tpn_i)
        constraint_zpn_rows.append(constraint_zpn_i)
        constraint_yp_rows.append(constraint_yp_i)
        constraint_yn_rows.append(constraint_yn_i)

    # Set constraint: t equals difference of objective value pairs row-wise (e.g. t0102 - x01 + x02 = 0)
    t_cols = []
    constraint_t_cols = []
    for i in range(0, board_len):
        t_x = []
        constraint_t_x = []
        for k in range(0, board_len-1):
            t_y = []
            constraint_t_y = []
            for j in range(0 + k, board_len-1):
                # one t for each unique objective variable x pair
                t_y.append(
                    solver.IntVar(
                        -10, 10, 't_'+str(k)+str(i)+str(j+1)+str(i)))  # range of -10 to 10
                constraint_t_y.append(
                    solver.Add(  # e.g. t0001=x00-x01 ... t0002=x00-x02
                        t_y[j-k]
                        - objective_vars[k][i]
                        + objective_vars[j+1][i]
                        == 0,
                        't_' + str(k) + str(i) + str(j+1) + str(i)))
                #print("Constraint: " + t_y[j-k].name() + ' - ' + objective_vars[k][i].name()
                #        + ' + ' + objective_vars[j+1][i].name() + ' = ' + ' 0 ')
            t_x.append(t_y)
            constraint_t_x.append(constraint_t_y)
        t_cols.append(t_x)
        constraint_t_cols.append(constraint_t_x)

    # Create constraint variables p, n, z, and y
    p_cols = []
    n_cols = []
    z_cols = []
    y_cols = []
    for row in t_cols:
        p_x = []
        n_x = []
        z_x = []
        y_x = []
        for column in row:
            p_y = []
            n_y = []
            z_y = []
            y_y = []
            for variable_t in column:
                p_y.append(
                    solver.IntVar(
                        0, 10, 'p_' + variable_t.name()[2:6]))
                n_y.append(
                    solver.IntVar(
                        0, 10, 'n_' + variable_t.name()[2:6]))
                z_y.append(
                    solver.IntVar(
                        1, 10, 'z_' + variable_t.name()[2:6]))  # Note that z must be greater than or equal one
                y_y.append(
                    solver.BoolVar(
                        'y_' + variable_t.name()[2:6]))
            p_x.append(p_y)
            n_x.append(n_y)
            z_x.append(z_y)
            y_x.append(y_y)
        p_cols.append(p_x)
        n_cols.append(n_x)
        z_cols.append(z_x)
        y_cols.append(y_x)

    # Set constraints: z equal to the absolute value of objective variable pair differences (Big M formulation)
    constraint_tpn_cols = []  # t-p+n=0
    constraint_zpn_cols = []  # z-p-n=0     e.g. z is the absolute value of objective variable pair differences
    constraint_yp_cols = []  # p-10*y<=0   Big-M formulation, where M=10, to enforce either y=0 or p=0
    constraint_yn_cols = []  # n+10*y<=10  Big-M formulation, where M=10, to enforce either y=0 or p=0
    for i in range(0, t_cols.__len__()):
        constraint_tpn_x = []
        constraint_zpn_x = []
        constraint_yp_x = []
        constraint_yn_x = []
        for j in range(0, t_cols[i].__len__()):
            constraint_tpn_y = []
            constraint_zpn_y = []
            constraint_yp_y = []
            constraint_yn_y = []
            for k in range(0, t_cols[i][j].__len__()):
                constraint_tpn_y.append(
                    solver.Add(  # t-p+n=0
                        t_cols[i][j][k]
                        - p_cols[i][j][k]
                        + n_cols[i][j][k]
                        == 0,
                        'tpn_' + t_cols[i][j][k].name()[2:6]))
                constraint_zpn_y.append(
                    solver.Add(  # z-p-n=0
                        z_cols[i][j][k]
                        - p_cols[i][j][k]
                        - n_cols[i][j][k]
                        == 0,
                        'zpn_' + z_cols[i][j][k].name()[2:6]))
                constraint_yp_y.append(  # p-10*y<=0
                    solver.Add(
                        p_cols[i][j][k]
                        - 10 * y_cols[i][j][k]
                        <= 0,
                        'yp_' + p_cols[i][j][k].name()[2:6]))
                constraint_yn_y.append(  # n+10*y<=10
                    solver.Add(
                        n_cols[i][j][k]
                        + 10 * y_cols[i][j][k]
                        <= 10,
                        'yn_' + n_cols[i][j][k].name()[2:6]))
            constraint_tpn_x.append(constraint_tpn_y)
            constraint_zpn_x.append(constraint_zpn_y)
            constraint_yp_x.append(constraint_yp_y)
            constraint_yn_x.append(constraint_yn_y)
        constraint_tpn_cols.append(constraint_tpn_x)
        constraint_zpn_cols.append(constraint_zpn_x)
        constraint_yp_cols.append(constraint_yp_x)
        constraint_yn_cols.append(constraint_yn_x)

    # Solution
    solver.SetTimeLimit(max_solve_time)
    status = solver.Solve()
    print(status)
    if status == 0:
        print("Success!")
    if status != 0:
        print("Solve failed, see status for details.")
    solution_values = np.array([
        [objective_vars[i][j].solution_value() for j in range(0, board_len)]
        for i in range(0, board_len)])
    model_as_string = solver.ExportModelAsLpFormat(False)

    # End timer
    end_time = datetime.now()
    run_time = end_time-start_time

    # More messaging
    if status == 0:
        print("Board of size " + str(board_len) + " solved in "
              + str(run_time.seconds) + " seconds, using " + str(solver.iterations())
              + " simplex iterations.")

    # Output
    return({
        "status": status,
        "solution": solution_values.astype('int'),
        "runtime_seconds": run_time.seconds,
        "lp_file": model_as_string,
        "solver": solver
    })

# Define ILP sudoku solver using COIN-CBC
def solve_board_COIN(initial_board, max_solve_time=120000):
    # libraries
    from ortools.linear_solver import pywraplp
    import numpy as np
    from datetime import datetime

    # Start timer
    start_time = datetime.now()

    # Create solver
    print("(1) Initializing optimization model...")
    solver = pywraplp.Solver(
        "Sudoku",
        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    objective = solver.Objective()
    objective.SetMinimization

    # Get needed range
    board_len = initial_board.__len__()

    # Create objective variable array
    print("(2) Creating objective variables...")
    objective_vars = np.array([
        [solver.IntVar(1, board_len, 'x' + str(j) + str(i)) for i in range(0, board_len)]
        for j in range(0, board_len)])

    # Set initial board constraints
    print("(3) Setting constraints...")
    constraint_initial = []
    for i in range(0, board_len):
        for j in range(0, board_len):
            if initial_board[i][j] != 0:
                constraint_initial.append(
                    solver.Add(
                        objective_vars[i][j] == initial_board[i][j],
                        'init_x'+str(i)+str(j)+'=='+str(initial_board[i][j])))
            else:
                constraint_initial.append(
                    'init_x'+str(i)+str(j)+'==NONE'
                )

    # Set constraint: sum of rows must equal 45 (Not necessary or sufficient, but could help bound.)
    constraint_rowsum = [[]] * board_len
    for i in range(0, board_len):
        constraint_rowsum[i] = solver.Add(
            sum(objective_vars[i]) == sum(range(board_len+1)))

    # Set constraint: sum of columns must equal 45
    constraint_colsum = [[]] * board_len
    for j in range(0, board_len):
        constraint_colsum[j] = solver.Add(
            sum(objective_vars[:, j]) == sum(range(board_len+1)))

    # Set constraint: nonet sums
    # constraint_nonet = []
    if board_len == 9:
        solver.Add(
            sum(sum(objective_vars[0:3, 0:3])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[0:3, 3:6])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[0:3, 6:9])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[3:6, 0:3])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[3:6, 3:6])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[3:6, 6:9])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[6:9, 0:3])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[6:9, 3:6])) == sum(range(0, board_len + 1)))
        solver.Add(
            sum(sum(objective_vars[6:9, 6:9])) == sum(range(0, board_len + 1)))


    # Set constraint: t equals difference of objective value pairs row-wise (e.g. t0102 - x01 + x02 = 0)
    t_rows = []
    constraint_t_rows = []
    for i in range(0, board_len):
        t_i = []
        constraint_t_i = []
        for k in range(0, board_len-1):
            t_j = []
            constraint_t_j = []
            for j in range(0+k, board_len-1):
                # one t for each unique objective variable x pair
                t_j.append(
                    solver.IntVar(
                        -10, 10, 't'+str(i)+str(k)+str(i)+str(j+1)))     # range of -10 to 10
                constraint_t_j.append(
                    solver.Add(     # e.g. t0001=x00-x01 ... t0002=x00-x02
                        t_j[j-k]
                        - objective_vars[i][k]
                        + objective_vars[i][j+1]
                        == 0,
                        't'+str(i)+str(k)+str(i)+str(j+1)))
                #print("Constraint: " + constraint_var_t_j[j-k].name() + ' - ' + objective_vars[i][k].name()
                #        + ' + ' + objective_vars[i][j+1].name() + ' = ' + ' 0 ')
            t_i.append(t_j)
            constraint_t_i.append(constraint_t_j)
        t_rows.append(t_i)
        constraint_t_rows.append(constraint_t_i)

    # Create constraint variables p, n, z, and y
    p_rows = []
    n_rows = []
    z_rows = []
    y_rows = []
    for row in t_rows:
        p_i = []
        n_i = []
        z_i = []
        y_i = []
        for column in row:
            p_j = []
            n_j = []
            z_j = []
            y_j = []
            for variable_t in column:
                p_j.append(
                    solver.IntVar(
                        0, 10, 'p'+variable_t.name()[1:5]))
                n_j.append(
                    solver.IntVar(
                        0, 10, 'n'+variable_t.name()[1:5]))
                z_j.append(
                    solver.IntVar(
                        1, 10, 'z'+variable_t.name()[1:5])) # Note that z must be greater than or equal one
                y_j.append(
                    solver.BoolVar(
                        'y'+variable_t.name()[1:5]))
            p_i.append(p_j)
            n_i.append(n_j)
            z_i.append(z_j)
            y_i.append(y_j)
        p_rows.append(p_i)
        n_rows.append(n_i)
        z_rows.append(z_i)
        y_rows.append(y_i)

    # Set constraints: z equal to the absolute value of objective variable pair differences (Big M formulation)
    constraint_tpn_rows = []     # t-p+n=0
    constraint_zpn_rows = []     # z-p-n=0     e.g. z is the absolute value of objective variable pair differences
    constraint_yp_rows = []      # p-10*y<=0   Big-M formulation, where M=10, to enforce either y=0 or p=0
    constraint_yn_rows = []      # n+10*y<=10  Big-M formulation, where M=10, to enforce either y=0 or p=0
    for i in range(0, t_rows.__len__()):
        constraint_tpn_i = []
        constraint_zpn_i = []
        constraint_yp_i = []
        constraint_yn_i = []
        for j in range(0, t_rows[i].__len__()):
            constraint_tpn_j = []
            constraint_zpn_j = []
            constraint_yp_j = []
            constraint_yn_j = []
            for k in range(0, t_rows[i][j].__len__()):
                constraint_tpn_j.append(
                    solver.Add(     # t-p+n=0
                        t_rows[i][j][k]
                        - p_rows[i][j][k]
                        + n_rows[i][j][k]
                        == 0,
                        'tpn'+t_rows[i][j][k].name()[1:5]))
                constraint_zpn_j.append(
                    solver.Add(     # z-p-n=0
                        z_rows[i][j][k]
                        - p_rows[i][j][k]
                        - n_rows[i][j][k]
                        == 0,
                        'zpn'+z_rows[i][j][k].name()[1:5]))
                constraint_yp_j.append(     # p-10*y<=0
                    solver.Add(
                        p_rows[i][j][k]
                        - 10*y_rows[i][j][k]
                        <= 0,
                        'yp'+p_rows[i][j][k].name()[1:5]))
                constraint_yn_j.append(     # n+10*y<=10
                    solver.Add(
                        n_rows[i][j][k]
                        + 10*y_rows[i][j][k]
                        <= 10,
                        'yn' + n_rows[i][j][k].name()[1:5]))
            constraint_tpn_i.append(constraint_tpn_j)
            constraint_zpn_i.append(constraint_zpn_j)
            constraint_yp_i.append(constraint_yp_j)
            constraint_yn_i.append(constraint_yn_j)
        constraint_tpn_rows.append(constraint_tpn_i)
        constraint_zpn_rows.append(constraint_zpn_i)
        constraint_yp_rows.append(constraint_yp_i)
        constraint_yn_rows.append(constraint_yn_i)

    # Set constraint: t equals difference of objective value pairs row-wise (e.g. t0102 - x01 + x02 = 0)
    t_cols = []
    constraint_t_cols = []
    for i in range(0, board_len):
        t_x = []
        constraint_t_x = []
        for k in range(0, board_len-1):
            t_y = []
            constraint_t_y = []
            for j in range(0 + k, board_len-1):
                # one t for each unique objective variable x pair
                t_y.append(
                    solver.IntVar(
                        -10, 10, 't_'+str(k)+str(i)+str(j+1)+str(i)))  # range of -10 to 10
                constraint_t_y.append(
                    solver.Add(  # e.g. t0001=x00-x01 ... t0002=x00-x02
                        t_y[j-k]
                        - objective_vars[k][i]
                        + objective_vars[j+1][i]
                        == 0,
                        't_' + str(k) + str(i) + str(j+1) + str(i)))
                #print("Constraint: " + t_y[j-k].name() + ' - ' + objective_vars[k][i].name()
                #        + ' + ' + objective_vars[j+1][i].name() + ' = ' + ' 0 ')
            t_x.append(t_y)
            constraint_t_x.append(constraint_t_y)
        t_cols.append(t_x)
        constraint_t_cols.append(constraint_t_x)

    # Create constraint variables p, n, z, and y
    p_cols = []
    n_cols = []
    z_cols = []
    y_cols = []
    for row in t_cols:
        p_x = []
        n_x = []
        z_x = []
        y_x = []
        for column in row:
            p_y = []
            n_y = []
            z_y = []
            y_y = []
            for variable_t in column:
                p_y.append(
                    solver.IntVar(
                        0, 10, 'p_' + variable_t.name()[2:6]))
                n_y.append(
                    solver.IntVar(
                        0, 10, 'n_' + variable_t.name()[2:6]))
                z_y.append(
                    solver.IntVar(
                        1, 10, 'z_' + variable_t.name()[2:6]))  # Note that z must be greater than or equal one
                y_y.append(
                    solver.BoolVar(
                        'y_' + variable_t.name()[2:6]))
            p_x.append(p_y)
            n_x.append(n_y)
            z_x.append(z_y)
            y_x.append(y_y)
        p_cols.append(p_x)
        n_cols.append(n_x)
        z_cols.append(z_x)
        y_cols.append(y_x)

    # Set constraints: z equal to the absolute value of objective variable pair differences (Big M formulation)
    constraint_tpn_cols = []  # t-p+n=0
    constraint_zpn_cols = []  # z-p-n=0     e.g. z is the absolute value of objective variable pair differences
    constraint_yp_cols = []  # p-10*y<=0   Big-M formulation, where M=10, to enforce either y=0 or p=0
    constraint_yn_cols = []  # n+10*y<=10  Big-M formulation, where M=10, to enforce either y=0 or p=0
    for i in range(0, t_cols.__len__()):
        constraint_tpn_x = []
        constraint_zpn_x = []
        constraint_yp_x = []
        constraint_yn_x = []
        for j in range(0, t_cols[i].__len__()):
            constraint_tpn_y = []
            constraint_zpn_y = []
            constraint_yp_y = []
            constraint_yn_y = []
            for k in range(0, t_cols[i][j].__len__()):
                constraint_tpn_y.append(
                    solver.Add(  # t-p+n=0
                        t_cols[i][j][k]
                        - p_cols[i][j][k]
                        + n_cols[i][j][k]
                        == 0,
                        'tpn_' + t_cols[i][j][k].name()[2:6]))
                constraint_zpn_y.append(
                    solver.Add(  # z-p-n=0
                        z_cols[i][j][k]
                        - p_cols[i][j][k]
                        - n_cols[i][j][k]
                        == 0,
                        'zpn_' + z_cols[i][j][k].name()[2:6]))
                constraint_yp_y.append(  # p-10*y<=0
                    solver.Add(
                        p_cols[i][j][k]
                        - 10 * y_cols[i][j][k]
                        <= 0,
                        'yp_' + p_cols[i][j][k].name()[2:6]))
                constraint_yn_y.append(  # n+10*y<=10
                    solver.Add(
                        n_cols[i][j][k]
                        + 10 * y_cols[i][j][k]
                        <= 10,
                        'yn_' + n_cols[i][j][k].name()[2:6]))
            constraint_tpn_x.append(constraint_tpn_y)
            constraint_zpn_x.append(constraint_zpn_y)
            constraint_yp_x.append(constraint_yp_y)
            constraint_yn_x.append(constraint_yn_y)
        constraint_tpn_cols.append(constraint_tpn_x)
        constraint_zpn_cols.append(constraint_zpn_x)
        constraint_yp_cols.append(constraint_yp_x)
        constraint_yn_cols.append(constraint_yn_x)

    # Solution
    solver.SetTimeLimit(max_solve_time)
    status = solver.Solve()
    print(status)
    if status == 0:
        print("Success!")
    if status != 0:
        print("Solve failed, see status for details.")
    solution_values = np.array([
        [objective_vars[i][j].solution_value() for j in range(0, board_len)]
        for i in range(0, board_len)])
    model_as_string = solver.ExportModelAsLpFormat(False)

    # End timer
    end_time = datetime.now()
    run_time = end_time-start_time

    # More messaging
    if status == 0:
        print("Board of size " + str(board_len) + " solved in "
              + str(run_time.seconds) + " seconds, using " + str(solver.iterations())
              + " simplex iterations.")

    # Output
    return({
        "status": status,
        "solution": solution_values.astype('int'),
        "runtime_seconds": run_time.seconds,
        "lp_file": model_as_string,
        "solver": solver
    })

# Define sudoku board
difficult_board = np.array([
    [2, 0, 0, 3, 0, 0, 0, 0, 0],
    [8, 0, 4, 0, 6, 2, 0, 0, 3],
    [0, 1, 3, 8, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 2, 0, 3, 9, 0],
    [5, 0, 7, 0, 0, 0, 6, 2, 1],
    [0, 3, 2, 0, 0, 6, 0, 0, 0],
    [0, 2, 0, 0, 0, 9, 1, 4, 0],
    [6, 0, 1, 2, 5, 0, 8, 0, 9],
    [0, 0, 0, 0, 0, 1, 0, 0, 2]
])

# Solve with COIN-CBC
board_solution_COIN = solve_board_COIN(difficult_board, max_solve_time = 60000)
print(board_solution_COIN['solution']) # no solution found in 60 seconds

# Solve with CPLEX
board_solution_CPLEX = solve_board_CPLEX(difficult_board, max_solve_time = 60000)
print(board_solution_CPLEX['solution']) # success - solution found