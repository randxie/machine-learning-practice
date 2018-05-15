import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

COMPUTE_OP = ['+', '-', '*', '/']
OPERATORS = ['+', '-', '*', '/', '(', ')']

def compare_op(op1, op2):
    # compare precedence of op1 and op2
    if op2 in ['+', '-']:
        if op1 in ['*', '/']:
            return True
    elif op2 == '(':
        return True

    return False


def infix_to_postfix_conversion(infix_expr):
    """
    Follow descriptions in http://interactivepython.org/runestone/static/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html
    """
    tmp_expr = ''.join(map(lambda x: x if x not in OPERATORS else (' ' + x + ' '), infix_expr)).split()

    opstack = []
    output_list = []

    for w in tmp_expr:
        if w in COMPUTE_OP:
            while len(opstack) > 0 and (not compare_op(w, opstack[-1])):
                tmp_op = opstack.pop()
                output_list.append(tmp_op)
            opstack.append(w)
        elif w == '(':
            opstack.append(w)
        elif w == ')':
            while len(opstack) > 0 and opstack[-1] != '(':
                tmp_op = opstack.pop()
                output_list.append(tmp_op)
            opstack.pop()
        else:
            output_list.append(w)

    output_list.extend(opstack[::-1])
    return output_list

def construct_expression_tree(postfix_expr):
    """
    Refer to https://cs.nyu.edu/courses/fall11/CSCI-GA.1133-001/rct257_files/Expression_Trees.pdf
    """
    # to be continued
    G = nx.DiGraph()

    root = postfix_expr.pop()
    G.add_node(root)
    cur_node = root
    while len(postfix_expr) > 0:
        node_to_add = postfix_expr.pop()
        while node_to_add in G:
            node_to_add = ':' + node_to_add

        while len(G[cur_node]) >= 2:
            cur_node = G.node[cur_node]['parent']

        G.add_edge(cur_node, node_to_add)
        G.node[node_to_add]['parent'] = cur_node

        if any(op in node_to_add for op in OPERATORS):
            cur_node = node_to_add

    return G

if __name__ == '__main__':
    postfix_expr = infix_to_postfix_conversion('(8 -5) * ((4+2)/3)')
    print(postfix_expr)
    expression_tree = construct_expression_tree(postfix_expr)
    plt.title('Expression Tree')
    pos = graphviz_layout(expression_tree, prog='dot')
    nx.draw(expression_tree, pos, with_labels=True, arrows=True)
    plt.show()