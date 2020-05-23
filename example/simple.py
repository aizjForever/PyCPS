def insert(p):
    elt, l = p
    if l == []:
        return [elt]

    if elt <= l[0]:
        return [elt] + l
    else:
        return [l[0]] + insert((elt, l[1:]))

def isort(l):
    if l == []:
        return l
    else:
        first = l[0]
        ll = isort(l[1:])
        return insert((first, ll))

def fib(n):
    if n == 0 or n == 1: return 1
    else: return fib(n - 1) + fib(n - 2)

def add(t):
    a, b = t
    if b == 0: return a
    else: return add((a, b - 1)) + 1


