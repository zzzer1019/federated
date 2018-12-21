<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.Model" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="local_variables"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="aggregated_outputs"/>
<meta itemprop="property" content="forward_pass"/>
</div>

# tff.learning.Model

## Class `Model`



Represents a Model for use in TensorFlow Federated.

Each Model will work on a set of Variables, and each method should be
a computation that can be implemented as a `tf.defun`; this implies the class
should essentially be stateless from a Python perspective, as each method
will generally only be traced once (per set of arguments) to create the
corresponding TensorFlow graph functions. Thus, `Model` instances should
behave as expected in both eager and graph (TF 1.0) usage.

In general, Variables may be either:
  - model variables, that are needed to make predictions with the model
  - non-model local variables, e.g. to accumulate aggregated metrics across
    calls to forward_pass.
Model variables can be broken down into trainable variables (variables
that can and should be trained using gradient-based methods), and
non-trainable variables (which could include fixed pre-trained layers,
or static model data). These variables are provided via the
`trainable_variables`, `non_trainable_variables`, and `local_variables`
properties, and must be initialized by the user of the `Model`.

In federated learning, model variables will generally be provided by the
server, and updates to trainable model variables will be sent back to the
server. Local variables are not transmitted, and are instead initialized
locally on the device, and then used to produce `aggregated_outputs` which
are sent to the server.

All `tf.Variables` should be introduced in `__init__`; this could move to a
`build()` method more inline with Keras (see
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) in
the future.

## Properties

<h3 id="local_variables"><code>local_variables</code></h3>

An iterable of `tf.Variable` objects, see class comment for details.

<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>

An iterable of `tf.Variable` objects, see class comment for details.

<h3 id="trainable_variables"><code>trainable_variables</code></h3>

An iterable of `tf.Variable` objects, see class comment for details.



## Methods

<h3 id="aggregated_outputs"><code>aggregated_outputs</code></h3>

``` python
aggregated_outputs()
```

Returns tensors representing values aggregated over forward_pass calls.

In federated learning, the values returned by this method will typically
be further aggregated across clients and made available on the server.

TODO(b/120147094): Support specification of aggregation across clients.

This method returns results from aggregating across *all* previous calls
to `forward_pass`, most typically metrics like Accuracy and Loss. If needed,
we may add a `clear_aggregated_outputs` method, which would likely just
run the initializers on the `local_variables`.

In general, the tensors returned can be an arbitrary function of all
the Variables of this model, not just the `local_variables`; for example,
a this could return tensors measuring the total L2 norm of the model
(which might have been updated by training).

This method may return arbitrarily shaped tensors, not just scalar metrics.
For example, it could return the average feature vector or a count of
how many times each feature exceed a certain magnitude.

#### Returns:

A structure of tensors (as supported by tf.contrib.framework.nest)
to be aggregated across clients.

<h3 id="forward_pass"><code>forward_pass</code></h3>

``` python
forward_pass(
    batch,
    training=True
)
```

Runs the forward pass and returns results.

This method should not modify any variables that are part of the model,
that is, variables that influence the predictions; for that, see
`TrainableModel.train_on_batch`.

However, this method may update aggregated metrics computed across calls to
forward_pass; the final values of such metrics can be accessed via
`aggregated_outputs`.

Uses in TFF:
  - To implement model evaluation.
  - To implement federated gradient descent and other
    non-FederatedAvgeraging algorithms, where we want the model to run the
    forward pass and update metrics, but there is no optimizer
    (we might only compute gradients on the returned loss).
  - To implement FederatedAveraging, when augmented as a `TrainableModel`.

TODO(b/120493676): We expect to add another method to this class which
provides access to the shape/dtype/structure expected for the `batch`.

#### Args:

* <b>`batch`</b>: A structure of tensors (as supported by tf.contrib.framework.nest,
    or could be produced by a `tf.data.Dataset`) for the current batch. It
    is the caller's responsibility to provide data of the format expected by
    the Model being called.
* <b>`training`</b>: If True, run the training forward pass, otherwise, run in
    evaluation mode. The semantics are generally the same as the `training`
    argument to `keras.Model.__call__`; this might e.g. influence how
    dropout or batch normalization is handled.


#### Returns:

A BatchOutput namedtuple. This must define a `loss` tensor if the model
will be trained via a gradient-based algorithm.


