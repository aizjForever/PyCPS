#variable
class Variable(object):
    count = 0
    def __init__(self, count):
        self.count = count

    @classmethod
    def newvar(cls):
        count = cls.count
        cls.count += 1
        return cls(count)


    def __repr__(self):
        return f"v{self.count}"

    def __eq__(self, other):
        return self.count == other.count
 