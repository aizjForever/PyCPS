# PyCPS
A CPS conversion tool for Python 3

## Requirements

The [astor](https://astor.readthedocs.io/en/latest/) python package
To install the requirements, please use `pip3`:  
```
pip3 install -r requirements.txt
```

## To Do

1. Multiple function call arguments
2. additional pass: fix the scope of variables and declare global when necessary
3. different lazy eval strategy: yield vs. lambda (): ...
4. In trans_while, need to figure out what nonlocal variables are used in order to update 
5. the defaults for the keyword args (done in additional pass)
6. Since we translate while loops to tail calls, return statement in the source while might cause problems
