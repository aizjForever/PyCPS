# PyCPS
A CPS conversion tool for Python 3

## Requirements

The [astor](https://astor.readthedocs.io/en/latest/) python package
To install the requirements, please use `pip3`:  
```
pip3 install -r requirements.txt
```

## To Do

* additional pass: fix the scope of variables and declare global when necessary
* different lazy eval strategy: yield vs. lambda (): ...
* In trans_while, need to figure out what nonlocal variables are used in order to update the defaults for the keyword args (done in additional pass)
* Since we translate while loops to tail calls, return statements in the source while might cause problems
