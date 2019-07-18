from plugins.admin.Petal import Petal
from yapsy.PluginManager import PluginManager
import numpy as np


class PetalManager(PluginManager):
    """
    """
    petals = []
    id = np.random.randint(0x00, 0xFFFFFFFF)

    def __init__(self, path_to_plugins=None):
        super(PetalManager, self).__init__()

        self.setPluginPlaces(path_to_plugins)

        self.setCategoriesFilter(
            {
                "Default": Petal,
                #"RandomGenerator": RandomGenerator
            }
        )

        self.collectPlugins()
        self.paths = path_to_plugins

    @property
    def activated_petals(self):
        return self.petals

    def load_petals(self, **kwargs):
        try:
            desired_petals = kwargs['petals']
        except KeyError:
            desired_petals = [petal.name for petal in self.getAllPlugins()]

        for petal in desired_petals:
            petal_info = self.getPluginByName(petal)
            petal_info.plugin_object.parent = self
            self.activatePluginByName(petal_info.name)
            self.petals.append(petal_info)

    def get_petals(self, **kwargs):
        try:
            petals = kwargs['petals']
            if len(petals) == 1:
                petal = self.getPluginByName(kwargs['petals'][0])

                if not petal.is_activated:
                    # petal is not activated, and can not load
                    return None
                return petal

            else:
                petals = [petal if petal.is_activated else None
                          for petal in
                          [self.getPluginByName(petal) for petal in petals]
                          ]

                if None in petals:
                    print("Warning - Some petals are not activated, skipping them.")
                    return list(filter(None, petals))

                return petals
        except KeyError:
            print("No such petal found, KeyError")
            return None

    def get_petal(self, **kwargs):
        try:
            petal = self.getPluginByName(kwargs['name'])
            if not petal.is_activated:
                print("petal not activated")
                return None

            return petal

        except KeyError:
            print("No such petal found, KeyError")
            return None

    def obj(self, name):
        petal_info = self.get_petal(name=name)
        return petal_info.plugin_object

    def run_petal(self, name, **kwargs):
        petal_obj = self.obj(name)
        petal_obj.run(kwargs)

    def unload_petals(self, **kwargs):

        try:
            desired_petals = kwargs['petals']
        except KeyError:
            desired_petals = self.petals

        for petal in desired_petals:
            self.deactivatePluginByName(petal.name)


if __name__ == "__main__":
    petal_manager = PetalManager(path_to_plugins=['./plugins'])
    petal_manager.load_petals(petals=['random'])
    random_plugin = petal_manager.get_petals(petals=['random'])
    print(random_plugin.plugin_object.run(amt=5))
