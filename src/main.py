import ast
from variable import Variable
from ast import *
import astor

class Unimplemented(Exception):
    pass

s = """
tally = 0
def f(x):
    if x:
        return x + f(x-1)
    return 0
    
    
"""
exe = """
identity = lambda x: x
def exec(f, x):
    val = f(x, identity)
    while callable(val):
        val = val()
    return val

# Note: to run your function, just run `exec(fun_name, arg)' v1.0 only supports function calls with one argument

"""


######
#
# To do: Multiple function arguments / assignment
#        Support more constructs
#        Need to insert return statements when necessary
#        advanced: need to keep track of the scope of variables and declare global when necessary
#
######



def transform(s):
    m = ast.parse(s)
    a = trans_mod(m)
    return astor.to_source(Module(body = a)) + exe

def make_arguments(args):
    return arguments(posonlyargs = [], args = args, vararg = None, kwonlyargs = [],
                     kw_defaults = [], kwarg= None,   defaults = [])

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
            assert len(args.args) == 1



            body = stmt.body

            s, k = trans_stmts(body)

            if k is None:
                k = Variable.newvar ()
           

            defn = FunctionDef(name = stmt.name, 
                               args = make_arguments(args.args + [make_arg(str(k))]),
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
            raise Unimplemented

    return s_accu, kres

def trans_exp_node(e, exps):
    assert len(exps)
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

    print (operands)
    print(astor.to_source(e))
    if s_accu is not None:
        return s_accu, kres, operands, b
    else:
        return e, None, None, None


def trans_exp(e):
    # e |-> s.k where s is a series of statements and k is a binding of the continuation 
    # Maybe should use symbol for k's
    if isinstance(e, BoolOp):
        op = e.op
        values = e.values
        s, k, operands, b = trans_exp_node(e, values)
        if k is None:
            return s, k
        else:
            b.args = [BoolOp(op, operands)]
            return s, k
        
    elif isinstance(e, BinOp):
        op = e.op
        exps = [e.left, e.right]
        s, k, operands, b = trans_exp_node(e, exps)
        if k is None:
            return s, k
        else:
            b.args = [BinOp(operands[0], op, operands[1])]
            return s, k


    elif isinstance(e, UnaryOp):
        op = e.op
        exps = [e.operand]
        s, k, operands, b = trans_exp_node(e, exps)
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
        params = e.args
        assert len(params) == 1
        [a] = params
        v1, f1 = trans_exp(func)
        v2, f2 = trans_exp(a)

        if f1 is None and f2 is None:
            k = Variable.newvar ()
            s = [Return(value = lazy_func_call(func = func, args = [a, Name(str(k), ctx = Load)], keywords = [])[0])]
            return s, k
        elif f1 is None and f2 is not None:
            f2n = Name(str(f2), Store)
            x = Variable.newvar ()
            k = Variable.newvar ()
            lam_body, _ = lazy_func_call(func = func, args = [Name(str(x), ctx = Load), Name(str(k), ctx = Load)], keywords = [])
            lam = Lambda(make_singleton_argument(str(x)), lam_body)
            s = [Assign([f2n], lam)] + v2
            return s, k

        elif f1 is not None and f2 is None:
            f1n = Name(str(f1), Store)
            f = Variable.newvar ()
            k = Variable.newvar ()
            lam_body, _ = lazy_func_call(func = Name(str(f), ctx = Load), args = [a, Name(str(k), ctx = Load)], keywords = [])
            lam = Lambda(make_singleton_argument(str(f)), lam_body)
            s = [Assign([f1n], lam)] + v1
            return s, k

        else:
            f1n = Name(str(f1), Store)
            f2n = Name(str(f2), Store)
            f = Variable.newvar ()
            x = Variable.newvar ()
            k = Variable.newvar ()
            lam_body, _ = lazy_func_call(func = Name(str(f), ctx = Load), args = [Name(str(x), ctx = Load), Name(str(k), ctx = Load)], keywords = [])
            lam = Lambda (make_singleton_argument(str(x)), lam_body)
            defn_body = [Assign([f2n], lam)] + v2
            defn = FunctionDef(name = str(f1n), args = make_singleton_argument(str(f)), 
                               body = defn_body, decorator_list = [], returns = None,
                               type_comment = None)
            s = [defn] + v1
            return s, k


    else:
        # default branch of trans_exp
        return e, None



