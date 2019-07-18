from plugins.admin.Petal import Petal
import h5py
import os


class Load(object):
    pass

class Save(object):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Storage(Petal):
    dset_name_counter = 0
    db_path = os.path.dirname(os.path.realpath(__file__))

    db_path_full = db_path+"/hd5py_data/"
    @staticmethod
    def strip_last(path):
        return str(path).split('/')[-1]

    def save(self, filename, dataset, name, **kwargs):
        filename = self.db_path_full + filename + ".hdf5"

        over_write_file = False
        if 'overwrite' in kwargs.keys():
            over_write_file = kwargs['overwrite']
            del kwargs['overwrite']

        with h5py.File(filename, 'a' if over_write_file is False else 'w') as f:
            attrs = None
            if 'attrs' in kwargs.keys():
                attrs = kwargs['attrs']
                del kwargs['attrs']

            try:
                del f[name]
                group = f.create_group(name=name)
            except KeyError:
                group = f.create_group(name=name)

            dset_name = Storage.strip_last(group.name)
            dset = group.create_dataset(name=dset_name, data=dataset, **kwargs)

            if attrs is not None:
                for attr in attrs.keys():
                    dset.attrs[attr] = attrs[attr]

    def load(self, filename, name=None, get_values=False):
        filename = self.db_path_full + filename + ".hdf5"

        f = h5py.File(filename, 'r')
        dset_name = Storage.strip_last(name)
        if name is None:
            return f['/']

        else:
            return f[name][dset_name] if get_values is False else f[name][dset_name][:]









