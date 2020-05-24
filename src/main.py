import ast
from variable import Variable
from ast import *
import astor

class Unimplemented(Exception):
    pass

EXEC = """
identity = lambda x: x
def exec(f, *args):
    val = f(identity, *args)
    while callable(val):
        val = val()
    return val

# Note: to run your function, just run `exec(fun_name, arg)' v1.0 only supports function calls with one argument
# To call a function with multiple arguments, wrap the arguments in a tuple and unwrap in the beginning of the function
"""


######
#
# To do: Multiple function call arguments
#        additional pass: fix the scope of variables and declare global when necessary
#        different lazy eval strategy: yield vs. lambda (): ...
#        In trans_while, need to figure out what nonlocal variables are used in order to update 
#        the defaults for the keyword args (done in additional pass)
#
######



def transform(s):
    m = ast.parse(s)
    a = trans_mod(m)
    return astor.to_source(Module(body = a)) + EXEC

def make_arguments(args):
    return arguments(posonlyargs = [], args = args, vararg = None, kwonlyargs = [],
                     kw_defaults = [], kwarg= None,   defaults = [])

def make_default_arguments(args, defaults):
    return arguments(posonlyargs = [], args = args, vararg = None, kwonlyargs = [],
                     kw_defaults = [], kwarg= None,   defaults = defaults)

def make_arg(a):
    return arg(arg = a, annotation = None, type_comment = None)


def make_singleton_argument(a):
    aa = make_arg(a)
    return make_arguments([aa])

def make_null_argument():
    return make_arguments([])

def lazy_func_call(*, func, args, keywords = [], lazy = True):
    c = Call(func = func, args = args, keywords = keywords)
    return Lambda(make_null_argument(), c) if lazy else c, c


def trans_mod(m):
    body = m.body
    s, k = trans_stmts(body)
    assert k is None
    return s

def base_stmts(s_accu):
    kres = Variable.newvar ()
    s_accu = [Return(value = lazy_func_call(func = Name(str(kres), ctx = Load), 
                     args = [], keywords = [])[0])] + s_accu 

    return kres, s_accu


def local_vars_exps(es):
    res = []
    for e in es:
        if isinstance(e, Name):
            res.append(e.id)
        elif isinstance(e, List):
            res.extend(local_vars_exps(e.elts))
        elif isinstance(e, Tuple):
            res.extend(local_vars_exps(e.elts))

    return res

def local_vars(stmts):
    # find the local vars in a sequence of statements
    vs = []
    for s in stmts:
        if isinstance(s, Assign):
            vs.extend(local_vars_exps(s.targets))
        else:
            continue

    return vs


def trans_while_stmts(stmts):
    res = []
    for s in stmts:
        if isinstance(s, While):
            res += trans_while(s)
        else:
            res.append(s)

    return res

def trans_while(w):
    # Returns a sequence of stmts
    # Turns a while statement into a function declaration
    # No actual CPS transformation performed

    # To-do: correctly set the default args
    assert isinstance(w, While)
    
    body = trans_while_stmts(w.body)
    orelse = trans_while_stmts(w.orelse)



    lvs = local_vars(body)
    lvs_as_args = [make_arg(lv) for lv in lvs]

    lvs_orelse = local_vars(orelse)
    print(lvs_orelse)

    while_f = Variable.newvar() 
    test = Variable.newvar ()

    if_body = body + [Return(value = Call(func = Name(str(while_f), ctx = Load), args = [], 
              keywords = [keyword(arg = str(test), value = w.test)] + [keyword(arg = lv, value = Name(lv, ctx = Load)) for lv in lvs]))]


    if_orelse = orelse + [Return(value = Tuple([Name(elt, ctx = Load) for elt in lvs_orelse], ctx = Load))]

    defn_body = [If(Name(str(test), ctx = Load), if_body, if_orelse)]
    defn = FunctionDef(name = str(while_f),
                       args = make_default_arguments([str(test)] + lvs_as_args, 
                             [w.test] + [Constant(value = None) for lv in lvs]),

                       body = defn_body,
                       decorator_list = [], returns = None, type_comment = None)

    s_while = [defn] + ([Assign([Tuple([Name(x, ctx = Store) for x in lvs_orelse], ctx = Store)], 
                                Call(func = Name(str(while_f), ctx = Load), args = [], keywords = []))] if len(lvs_orelse) else [])

    return s_while





def trans_stmts(stmts, s_accu = [], kres = None):
    # represents a scope
    for stmt in stmts[::-1]:
        if isinstance(stmt, Return):
            # discard the rest as the rest is dead code
            val = stmt.value
            kres = Variable.newvar ()
            if val is None:
                
                s_accu = [Return(value = lazy_func_call(func = Name(str(kres), ctx = Load), 
                            args = [Constant(value = None, kind = None)], 
                            keywords = [])[0])]

            else:
                s, k = trans_exp(val)
                if k is None:
                     s_accu = [Return(value = lazy_func_call(func = Name(str(kres), ctx = Load), 
                                                             args = [s], keywords = [])[0])] 
                else:
                    s_accu = s
                    kres = k
                
        elif isinstance(stmt, FunctionDef):
            args = stmt.args


            body = stmt.body

            ktemp = Variable.newvar ()
            s, k = trans_stmts(body, 
                   s_accu = [Return(value = lazy_func_call(func = Name(str(ktemp), ctx = Load), 
                            args = [Constant(value = None, kind = None)], keywords = [])[0])], 
                   kres = ktemp) # always append a `return None' to function defns

            assert k is not None

            defn = FunctionDef(name = stmt.name, 
                               args = make_default_arguments([make_arg(str(k))] + args.args, stmt.args.defaults),
                               body = s,
                               decorator_list = [],
                               returns = None,
                               type_comment = None)

            s_accu = [defn] + s_accu 
            


        elif isinstance(stmt, Assign):
            assert len(stmt.targets) == 1 # only support single assignment for now

            # if kres is None and not no_cont:
            #     assert not len(s_accu)
            #     kres, s_accu = base_stmts(s_accu)

            target = stmt.targets[0]
            val = stmt.value

            s_tgt, k_tgt = trans_exp(target)
            s_val, k_val = trans_exp(val)
            

            if k_tgt is None and k_val is None:
                s_accu = [stmt] + s_accu

            elif k_tgt is not None and k_val is None:
                x = Variable.newvar ()
                defn = FunctionDef(name = str(k_tgt),
                                   args = make_singleton_argument(str(x)),
                                   body = [Assign([Name(str(x), ctx = Store)], s_val)] + s_accu,
                                   decorator_list = [], returns = None, type_comment = None
                                   )

                s_accu = [defn] + s_tgt

            elif k_tgt is None and k_val is not None:
                x = Variable.newvar ()
                defn = FunctionDef(name = str(k_val),
                                   args = make_singleton_argument(str(x)),
                                   body = [Assign([target], Name(str(x), ctx = Load))] + s_accu,
                                   decorator_list = [], returns = None, type_comment = None
                                   )

                s_accu = [defn] + s_val

            else:
                x = Variable.newvar ()
                y = Variable.newvar ()

                defn_val = FunctionDef(name = str(k_val),
                                       args = make_singleton_argument(str(y)),
                                       body = [Assign([Name(str(x), ctx = Store)], Name(str(y), ctx = Load))] + s_accu,
                                       decorator_list = [],
                                       returns = None,
                                       type_comment = None)

                defn_tgt = FunctionDef(name = str(k_tgt),
                                       args = make_singleton_argument(str(x)),
                                       body = [defn_val],
                                       decorator_list = [],
                                       returns = None,
                                       type_comment = None)

                s_accu = [defn_tgt] + s_tgt



        elif isinstance(stmt, Expr):
            val = stmt.value
            s, k = trans_exp(val)
            if k is None:
                s_accu = [stmt] + s_accu
            else:
                x = Variable.newvar ()
                defn = FunctionDef(name = str(k),
                                   args = make_singleton_argument(str(x)),
                                   body = s_accu,
                                   decorator_list = [], returns = None, type_comment = None)
                s_accu = [defn] + s


        elif isinstance(stmt, If):
            test = stmt.test
            body = stmt.body
            orelse = stmt.orelse

            s1, k1 = trans_stmts(body, s_accu = s_accu, kres = kres)
            s2, k2 = trans_stmts(orelse, s_accu = s_accu, kres = kres)
            
            kk = kres
            if k1 is not None:
                kk = k1

            if k2 is not None:
                kk = k2

            pre = []
            if kk != k1 and kk is not None and k1 is not None:
                pre.append(Assign([Name(str(k1), ctx = Store)], Name(str(kk), ctx =  Load)))
            if kk != k2 and kk is not None and k2 is not None:
                pre.append(Assign([Name(str(k2), ctx = Store)], Name(str(kk), ctx =  Load)))

            kres = kk

            st, kt = trans_exp(test)
            
            if kt is None:
                s_accu = pre + [If(st, s1, s2)]
            else:
                x = Variable.newvar ()
                defn = FunctionDef(name = str(kt),
                                   args = make_singleton_argument(str(x)),
                                   body = pre + [If(Name(str(x), ctx = Load), s1, s2)],
                                   decorator_list = [], returns = None, type_comment = None)

                s_accu = [defn] + st

        else:
            s_accu = [stmt] + s_accu

    return s_accu, kres

def trans_slice(s):
    # return itself if no control flow switching
    if isinstance(s, Slice):
        pass

    elif isinstance(s, ExtSlice):
        pass

    elif isinstance(s, Index):
        value = s.value
        return trans_exp(value)

def trans_exp_node(e, exps):
    operands = [None for _ in exps]
    kres = Variable.newvar ()
    s_accu = None
    val = None
    for ii, exp in enumerate(exps[::-1]):
        i = len(exps) - ii - 1
        s, k = trans_exp(exp)
        if k is None:
            operands[i] = exp
        else:
            x = Variable.newvar ()
            operands[i] = Name(str(x), ctx = Load)

            if s_accu is None:
                kn = Name(str(k), ctx = Store)                
                bb, b = lazy_func_call(func = Name(str(kres), ctx = Load), args = [], keywords = [])
                lam = Lambda(make_singleton_argument(str(x)), bb)
                s_accu = [Assign([kn], lam)] + s

            else:
                s_accu = [FunctionDef(name = str(k), args = make_singleton_argument(str(x)),
                                     body = s_accu, decorator_list = [], returns = None, 
                                     type_comment = None)] + s

    if s_accu is not None:
        return s_accu, kres, operands, b, lam
    else:
        return e, None, None, None, None


def trans_exp(e):
    # e |-> s.k where s is a series of statements and k is a binding of the continuation 
    # Maybe should use symbol for k's
    if isinstance(e, BoolOp):
        op = e.op
        values = e.values
        s, k, operands, b, _ = trans_exp_node(e, values)
        if k is None:
            return s, k
        else:
            b.args = [BoolOp(op, operands)]
            return s, k
        
    elif isinstance(e, BinOp):
        op = e.op
        exps = [e.left, e.right]
        s, k, operands, b, _ = trans_exp_node(e, exps)
        if k is None:
            return s, k
        else:
            b.args = [BinOp(operands[0], op, operands[1])]
            return s, k


    elif isinstance(e, UnaryOp):
        op = e.op
        exps = [e.operand]
        s, k, operands, b, _ = trans_exp_node(e, exps)
        if k is None:
            return s, k
        else:
            b.args = [UnaryOp(op, operands[0])]
            return s, k

    elif isinstance(e, IfExp):
        test = e.test
        body = e.body
        orelse = e.orelse

        v2, k2 = trans_exp(body)
        v3, k3 = trans_exp(orelse)
        kres = k3

        if k2 is not None and k3 is not None:
            br_body = [Assign([Name(str(k2), ctx = Store)], Name(str(k3), ctx = Load))] + v2
            br_orelse = v3

        elif k2 is None and k3 is not None:
            br_body = [Return(value = lazy_func_call(func = Name(str(k3), ctx = Load), args = [v2], keywords = [])[0])]
            br_orelse = v3

        elif k2 is not None and k3 is None:
            br_body = v2
            br_orelse = [Return(value = lazy_func_call(func = Name(str(k2), ctx = Load), args = [v3], keywords = [])[0])]
            kres = k2

        else:
            kres = Variable.newvar ()
            br_body = [Return(value = lazy_func_call(func = Name(str(kres), ctx = Load), args = [v2], keywords = [])[0])]
            br_orelse = [Return(value = lazy_func_call(func = Name(str(kres), ctx = Load), args = [v3], keywords = [])[0])]

        v1, k1 = trans_exp(test)
        if k1 is None:
            sres = [If(v1, br_body, br_orelse)]
        else:
            t = Variable.newvar ()
            sres = [FunctionDef(name = str(k1), args = make_singleton_argument(str(t)),
                                body = [If(Name(str(t), ctx = Load), br_body, br_orelse)],
                                decorator_list = None, returns = None, type_comment = None)] + v1

        return sres, kres

    elif isinstance(e, Call):
        func = e.func
        args = e.args
        keyword_exps = [keyword.value for keyword in e.keywords]

        s, k, operands, b, lam = trans_exp_node(e, [func] + args + keyword_exps)

        if k is None:
            k = Variable.newvar ()
            return [Return(value=lazy_func_call(func = func, args = [Name(str(k), ctx = Load)] + e.args, keywords = e.keywords)[0])], k

        else:
            print("33")
            lam.body = lazy_func_call(func = operands[0], 
                                      args = [Name(str(k), ctx = Load)] + operands[1:len(e.args) + 1], 
                                      keywords = [keyword(a.arg, b) for a, b in zip(e.keywords, operands[len(e.args) + 1:])])[0]

            return s, k


    elif isinstance(e, Compare):
        left = e.left
        comparators = e.comparators
        ops = e.ops

        values = [left] + comparators
        s, k, operands, b, _ = trans_exp_node(e, values)

        if k is None:
            return s, k
        else:
            b.args = [Compare(left = operands[0], ops = ops, comparators = operands[1:])]
            return s, k

    elif isinstance(e, List):
        elts = e.elts
        ctx = e.ctx

        s, k, operands, b, _ = trans_exp_node(e, elts)
        if k is None:
            return s, k
        else:
            b.args = [List(elts = operands, ctx = ctx)]
            return s, k

    elif isinstance(e, Tuple):
        elts = e.elts
        ctx = e.ctx

        s, k, operands, b, _ = trans_exp_node(e, elts)
        if k is None:
            return s, k
        else:
            b.args = [Tuple(elts = operands, ctx = ctx)]
            return s, k

    elif isinstance(e, Attribute):
        value = e.value
        attr = e.attr
        ctx = e.ctx

        s, k, operands, b, _ = trans_exp_node(e, [value])
        if k is None:
            return s, k
        else:
            b.args = [Attribute(value = operands[0], attr = attr, ctx = ctx)]
            return s, k

    elif isinstance(e, Subscript):
        value = e.value
        sli = e.slice
        ctx = e.ctx

        s2, k2 = trans_exp(value)
        if isinstance(sli, Slice):
            val = []
            if sli.lower is not None:
                val.append(sli.lower)
            if sli.upper is not None:
                val.append(sli.upper)
            if sli.step is not None:
                val.append(sli.step)

            s, k, operands, b, _ = trans_exp_node(e, val)
            if k is None:
                if k2 is None:
                    return e, None
                else:
                    x = Variable.newvar ()
                    k = Variable.newvar ()
                    lam_body = lazy_func_call(func = Name(str(k), ctx = Load), 
                                              args = [Subscript(Name(str(x), ctx = Load), sli, ctx)], 
                                              keywords = [])[0]

                    lam = Lambda(make_singleton_argument(str(x)), lam_body)
                    s = [Assign([Name(str(k2), ctx = Store)], lam)] + s2
                    return s, k
            else:
                if sli.lower is not None:
                    lower = operands[0]
                    operands = operands[1:]
                else:
                    lower = None

                if sli.upper is not None:
                    upper = operands[0]
                    operands = operands[1:]
                else:
                    upper = None

                if sli.step is not None:
                    step = operands[0]
                    operands = operands[1:]
                else:
                    step = None

                if k2 is None:
                    b.args = [Subscript(s2, Slice(lower, upper, step), ctx)]

                    return s, k

                else:
                    x = Variable.newvar ()
                    b.args = [Subscript(Name(str(x), ctx = Load), Slice(lower, upper, step), ctx)]
                    defn = FunctionDef(name = str(k2),
                                       args = make_singleton_argument(str(x)), 
                                       body = s,
                                       decorator_list = [], returns = None, type_comment = None
                                       )
                    ss = [defn] + s2
                    return ss, k

        elif isinstance(sli, Index):
            val = sli.value
            s1, k1 = trans_exp(val)
            if k1 is None and k2 is None:
                return e, None
            elif k1 is not None and k2 is None:
                x = Variable.newvar ()
                k = Variable.newvar ()
                lam_body = lazy_func_call(func = Name(str(k), ctx = Load), 
                                          args = [Subscript(s2, Index(Name(str(x))), ctx)], 
                                          keywords = [])[0]

                lam = Lambda(make_singleton_argument(str(x)), lam_body)
                s = [Assign([Name(str(k1), ctx = Store)], lam)] + s1
                return s, k

            elif k1 is None and k2 is not None:
                x = Variable.newvar ()
                k = Variable.newvar ()
                lam_body = lazy_func_call(func = Name(str(k), ctx = Load), 
                                          args = [Subscript(Name(str(x), ctx = Load), sli, ctx)], 
                                          keywords = [])[0]

                lam = Lambda(make_singleton_argument(str(x)), lam_body)
                s = [Assign([Name(str(k2), ctx = Store)], lam)] + s2
                return s, k

            else:
                x = Variable.newvar ()
                y = Variable.newvar ()
                k = Variable.newvar ()
                lam_body = lazy_func_call(func = Name(str(k), ctx = Load), 
                                    args = [Subscript(Name(str(x), ctx = Load), Index(Name(str(y), ctx = Load)), ctx)], 
                                    keywords = [])[0]

                lam = Lambda(make_singleton_argument(str(y)), lam_body)
                defn_body = [Assign([Name(str(k1), ctx = Store)], lam)] + s1
                defn = FunctionDef(name = str(k2), args = make_singleton_argument(str(x)),
                                   body = defn_body,
                                   decorator_list = [], returns = None, type_comment = None)
                s = [defn] + s2

                return s, k


        else:
            # Extended slices
            raise Unimplemented

    elif isinstance(e, Dict):
        keys = e.keys
        values = e.values

        s, k, operands, b, _ = trans_exp_node(e, [i for t in zip(keys, values) for i in t])

        if k is None:
            return e, None
        else:

            b.args = [Dict(operands[0::2], operands[1::2])]
            return s, k

    elif isinstance(e, Set):
        elts = e.elts

        s, k, operands, b, _ = trans_exp_node(e, elts)
        if k is None:
            return e, None
        else:
            b.args = [Set(operands)]
            return s, k

    else:
        # default branch of trans_exp
        return e, None

if __name__ == "__main__":
    with open("../example/dict.py", "r") as f:
        print(transform(f.read()))
    






