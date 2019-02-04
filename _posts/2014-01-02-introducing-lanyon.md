---
layout: post
title: General purpose Bootstrap calculation in Python
---

Overall, Python has had some libraries for boostrap estimates:

- bootstrap
- pyboot

The shortcomings of those libraries include that they are restricted to a limited set of functions.

We propose a new package where the user can provide the function (or inform one from numpy and scipy).

All calculations are numba-accelerated, so expect a reasonably good performance. The library works

Seamlessly with all kinds of numpy arrays.

The boostrap approach can be considered conceptually simpler than other theoretical-heavy estimation procedures.

We can calculate a bootstrap sample mean by:

$$
\frac{1}{n} \sum_i y_i
$$

This function is basically represented by:

```python
def bootstrap(y, stfn, **kwargs):
	"""
	Arguments:
	y: Sample vector
	stfn: A statistical function to apply to the sample
	(mean, variance, etc)
	n:    The number of samples to draw. Defaults to 10000
	(...) extra arguments to stfn
	Returns:
	st: A vector of statistics calculated from it.
	"""
	n = kwargs.get('n', 1000)
	st = zeros(n)
	for i in range(n):
		n[i] = stfn(y, **kwargs...)
	return st
```

Let's try to perform a Boostrap analysis on a neuroimaging dataset, via a user-defined function.

```python

```

Overall, a reasonable performance.

