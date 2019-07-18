from numpy import random
from yapsy.IPlugin import IPlugin


class IPetal(IPlugin):
    _parent = None
    _id = random.randint(0x00, 0xffffffff)

    @property
    def id(self):
        return self._id

    @property
    def parent(self):
        return self._parent


    @parent.setter
    def parent(self, petal_manager):
        self._parent = petal_manager


class Petal(IPetal):

    def run(self, *args, **kwargs):
        pass
