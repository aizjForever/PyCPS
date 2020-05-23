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



