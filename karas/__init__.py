import pickle

from karas.version import __version__


def serialize(obj, filename):
    with open(filename, 'wb') as f:
        f.write(pickle.dumps(obj))


def deserialize(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def replace_type(key):
    return key.replace('scalar/', '') \
        .replace('images/', '')


class KeyEntry(object):
    mode = 'train'
    type = None
    key = ''

    def __init__(self, key):
        items = key.split('/')
        if items[0] not in ('train', 'test'):
            items.insert(0, 'train')

        self.mode = items[0]
        if items[1] in ('scalar', 'images', 'others'):
            self.type = items[1]
            self.key = '/'.join(items[2:])
        else:
            self.key = '/'.join(items[1:])

    def __repr__(self):
        return self.mode + ' ' + self.type + ' ' + self.key

    def __eq__(self, other):
        if self.mode != other.mode or self.key != other.key:
            return False
        if self.type is not None:
            return self.type == other.type
        return True


def compare_key(key, tag):
    return KeyEntry(key) == KeyEntry(tag)
