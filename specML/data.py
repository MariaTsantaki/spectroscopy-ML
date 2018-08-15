import pkg_resources
from specML.model_training import Data, Model


def get_data():
    fname = pkg_resources.resource_filename('specML', '/data/combined_spec.hdf')
    return Data(fname, scale=False)


def get_model():
    data = get_data()
    fname = pkg_resources.resource_filename('specML', '/FASMA_large_ML.pkl')
    return Model(data, load=True, fname=fname)
