from plugins.admin.Petal import Petal
import configparser as cp

class ConfigParser(Petal):

    _parser = None

    @property
    def raw_parser(self):
        return self._parser

    def run(self, *args, **kwargs):
        self._parser = cp.ConfigParser()
        self._parser.read(kwargs['path'])

    @property
    def version(self):
        return self._get_value("core", "version")

    def _get_value(self, section, attribute):
        return self._parser[section][attribute]


