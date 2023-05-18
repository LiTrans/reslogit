"""Core functionality definitions

"""
import numpy as np
import theano
import theano.tensor as T

from theano import shared

FLOATX = theano.config.floatX


def shared_dataset(df_x, df_y):
    """Function that loads elements into a shared dataset

    Args:
        df_x (pandas.DataFrame): a pandas dataframe array.
        df_y (pandas.DataFrame): a pandas dataframe array.

    Returns:
        theano.tensor.sharedvar.TensorSharedVariable: dataset stored in shared memory.
    """
    shared_df_x = shared(np.asarray(df_x, dtype=FLOATX), borrow=True)
    shared_df_y = shared(np.asarray(df_y, dtype=FLOATX), borrow=True)

    return shared_df_x, T.cast(shared_df_y, 'int32')
