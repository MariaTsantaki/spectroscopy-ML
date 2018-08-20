import pkg_resources
from specML.model_training import Data, Model


def get_data():
    fname = pkg_resources.resource_filename('specML', '/data/spec_ml.hdf')
    return Data(fname, scale=False, with_quadratic_terms=False)


def get_model(data=None):
    if data is None:
        data = get_data()
    fname = pkg_resources.resource_filename('specML', '/FASMA_large_ML.pkl')
    return Model(data, load=True, fname=fname)
