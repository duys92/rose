from plugins.admin.PetalManager import PetalManager
import numpy as np


class IRose(object):
    """
    Revolutionary Observing Synthetic Entity

    Let the journey begin.
    """
    _id = np.random.randint(0, 0xffffff)
    _petal_manager = None
    _plugin_path = "../plugins"
    _rose_ini = None
    _instances = {}

    active_petals = {}
    inactive_petals = []

    @classmethod
    def instance(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = cls(*args, **kwargs)
        return cls._instances[cls]

    @property
    def uid(self):
        return self._id

    @property
    def version(self):
        rose_core = self.obj('ini_parser')
        return rose_core.version

    def __init__(self, *args, **kwargs):

        self._petal_manager = PetalManager([self._plugin_path])
        self._petal_manager.load_petals()

        self._load_all()

    def obj(self, name):

        if not isinstance(name, list):
            obj = self.active_petals[name]
            return obj
        else:
            names = list(name)
            objs = []
            for name in names:
                print(name)
                objs.append(self.active_petals[name])

            return objs

    def _load_all(self):
        self._load_essentials()
        self._load_non_essentials()
        print("Successfully loaded plugins: {}".format(", ".join([p for p in self.active_petals.keys()])))

        if len(self.inactive_petals) > 0:
            print("Unable to load following plugins: {}".format(", ".join([p for p in self.inactive_petals])))

    def _load_essentials(self):
        self._load_petal(name='ini_parser', path="../config/rose.ini")
        self._load_petal(name='hd5py')
        self._load_petal(name='tensorflow')

    def _load_non_essentials(self):
        pass

    def _load_petal(self, name, **run_args):
        try:
            petal_info = self._petal_manager.get_petal(name=name)
            obj = self._petal_manager.obj(petal_info.name)
            obj.run(**run_args)
            self._add_petal(petal_info.name, obj)
        except AttributeError:
            self.inactive_petals.append(name)

    def _add_petal(self, name, obj):
        self.active_petals[name] = obj

    def _remove_petal(self, name):
        del self.active_petals[name]


