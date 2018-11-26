'''
Solve LP *relaxation* of bin packing problem using column generation
assume that item_sizes are integers
'''

import numpy as np
import gurobi as grb

EPSILON = 10.**-10


def main(item_sizes, bin_size):
    '''
    :param item_sizes: ndarray
    :param bin_size: int
    :return: solution
    '''

    num_items = item_sizes.size
    check_solution_exists(item_sizes, bin_size)

    m = grb.Model("master")
    m.setParam('OutputFlag', 0)

    # initialize using singleton patterns (i.e., one bin per item)
    patterns = [[0 if y != x else 1 for y in range(num_items)] for x in range(num_items)]

    # Generate list of constraints (only item requirements)
    constr_list = []
    for i in range(num_items):
        constr_list.append(m.addConstr(lhs=0,
                                       sense=grb.GRB.GREATER_EQUAL,
                                       rhs=1,
                                       name='constraint: item ' + str(i)))

    # Generate variables (non-negative constraints and through columns)
    var_list = []
    for pi, p in enumerate(patterns):
        temp_col = grb.Column(coeffs=[1.0]*1,
                              constrs=[constr_list[i] for i in range(num_items) if patterns[pi][i]==1])
        var_list.append(m.addVar(vtype='C',
                                 name='pattern%d' % len(var_list),
                                 # name='pattern: ' + str(p),
                                 obj=1.,
                                 lb=0.,
                                 column=temp_col))

    iter_num = 1
    while True:
        # Solve SmallPrimal
        m.optimize()
        # print_feedback(m, constr_list, num_items)

        # Save primal and dual variables
        sav_primal = m.getAttr('x')
        sav_dual = m.getAttr('pi')

        # Compute `values' of each item for column generation
        val, new_col = get_violation(item_sizes, bin_size, m.getAttr('Pi'))
        if val <= 1+EPSILON:
            break

        # Add column
        patterns.append(list(new_col))
        temp_col = grb.Column(coeffs=[1.0]*sum(new_col),
                              constrs=[constr_list[i] for i in range(num_items) if new_col[i]==1])
        var_list.append(m.addVar(vtype='C',
                                 # name='pattern: ' + str(list(new_col)),
                                 name='pattern%d' % len(var_list),
                                 obj=1.,
                                 lb=0.,
                                 column=temp_col))


        # Does not seeem to help, or warm start already usd
        for i in range(len(var_list)-1):
            var_list[i].setAttr('PStart', sav_primal[i])
        var_list[-1].setAttr('PStart', 0.)

        for i, constr in enumerate(constr_list):
            constr.setAttr('DStart', sav_dual[i])


        iter_num += 1
        print('iteration: %d'%iter_num)

    print_feedback(m, constr_list, num_items)


def get_violation(item_sizes, bin_size, item_values):
    """
    Find a new configuration using DP.
    We want the most `valuable' configuration.
    Note that item values are equal to the dual variables
    """

    num_items = item_sizes.size
    best_values = [0.] * (bin_size+1)
    best_configurations = [()] * (bin_size+1)
    for i in range(num_items):
        new_best_values = [0.] * (bin_size+1)
        new_best_configurations = [None] * (bin_size+1)
        item_size, item_value = item_sizes[i], item_values[i]
        for j in range(bin_size+1):
            if j - item_size < 0:
                new_best_values[j], new_best_configurations[j] = best_values[j], best_configurations[j] + (0,)
                continue

            best = max((best_values[j-item_size]+item_value, best_configurations[j-item_size] + (1,)),
                       (best_values[j], best_configurations[j] + (0,)))
            new_best_values[j] = best[0]
            new_best_configurations[j] = best[1]

        best_values, best_configurations = new_best_values, new_best_configurations

    return max(zip(best_values, best_configurations))


def print_feedback(m, constr_list, num_items):
    print_primal(m)
    print_dual(constr_list, num_items)
    print('Objective:', m.objVal)


def print_primal(m):
    print('--- Primals --- ')
    for v in m.getVars():
        print(v.varName, '---',
              v.getAttr('x'))


def print_dual(constr_list, num_items):
    print('--- Duals --- ')
    for j in range(num_items):
        print(constr_list[j].ConstrName, '---',
              constr_list[j].getAttr('Pi'))


def check_solution_exists(item_sizes, bin_size):
    if not np.all(item_sizes <= bin_size):
        raise Exception('No solution possible')


if __name__ == '__main__':
    # item_sizes = np.array([4, 8, 1, 4, 2, 1])
    import timeit
    item_sizes = np.array([4, 8, 1, 4, 2, 1] * 30)
    bin_size = 13

    def wrap():
        main(item_sizes, bin_size)

    print(timeit.timeit(wrap, number=1))

