import numpy as np
import jax
import jax.numpy as jnp
import statsmodels.tsa.stattools

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = jnp.array(values)
    quantiles = jnp.array(quantiles)
    if sample_weight is None:
        sample_weight = jnp.ones(len(values))
    sample_weight = jnp.array(sample_weight)
    assert jnp.all(quantiles >= 0) and jnp.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = jnp.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = jnp.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= jnp.sum(sample_weight)
    return jnp.interp(quantiles, weighted_quantiles, values)


def effective_sample_size(x, nlags=100):
    y = statsmodels.tsa.stattools.acf(x, nlags=nlags)
    z = y[(y>0).cumprod()==1].sum()
    return x.shape[0] / (1+2*(z-1))