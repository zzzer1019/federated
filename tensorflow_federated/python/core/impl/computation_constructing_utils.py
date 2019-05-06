# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library implementing reusable `computation_building_blocks` constructs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import type_utils


def construct_federated_getitem_call(arg, idx):
  """Constructs computation building block passing getitem to federated value.

  Args:
    arg: Instance of `computation_building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `idx`.
    idx: Index, instance of `int` or `slice` used to address the
      `computation_types.NamedTupleType` underlying `arg`.

  Returns:
    Returns a `computation_building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`, the result
    of applying or mapping the appropriate `__getitem__` function, as defined
    by `idx`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(idx, (int, slice))
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getitem_comp = construct_federated_getitem_comp(arg, idx)
  return construct_map_or_apply(getitem_comp, arg)


def construct_federated_getattr_call(arg, name):
  """Constructs computation building block passing getattr to federated value.

  Args:
    arg: Instance of `computation_building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `name`.
    name: String name to address the `computation_types.NamedTupleType`
      underlying `arg`.

  Returns:
    Returns a `computation_building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`,
    the result of applying or mapping the appropriate `__getattr__` function,
    as defined by `name`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getattr_comp = construct_federated_getattr_comp(arg, name)
  return construct_map_or_apply(getattr_comp, arg)


def construct_federated_setattr_call(federated_comp, name, value_comp):
  """Returns building block for `setattr(name, value_comp)` on `federated_comp`.

  Constructs an appropriate communication intrinsic (either `federated_map` or
  `federated_apply`) as well as a `computation_building_blocks.Lambda`
  representing setting the `name` attribute of `federated_comp`'s `member` to
  `value_comp`, and stitches these together in a call.

  Notice that `federated_comp`'s `member` must actually define a `name`
  attribute; this is enforced to avoid the need to worry about theplacement of a
  previously undefined name.

  Args:
    federated_comp: Instance of
      `computation_building_blocks.ComputationBuildingBlock` of type
      `computation_types.FederatedType`, with member of type
      `computation_types.NamedTupleType` whose attribute `name` we wish to set
      to `value_comp`.
    name: String name of the attribute we wish to overwrite in `federated_comp`.
    value_comp: Instance of
      `computation_building_blocks.ComputationBuildingBlock`, the value to
      assign to `federated_comp`'s `member`'s `name` attribute.

  Returns:
    Instance of `computation_building_blocks.ComputationBuildingBlock`
    representing `federated_comp` with its `member`'s `name` attribute set to
    `value`.
  """
  py_typecheck.check_type(federated_comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(value_comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(federated_comp.type_signature,
                          computation_types.FederatedType)
  py_typecheck.check_type(federated_comp.type_signature.member,
                          computation_types.NamedTupleType)
  named_tuple_type_signature = federated_comp.type_signature.member
  setattr_lambda = construct_named_tuple_setattr_lambda(
      named_tuple_type_signature, name, value_comp)
  return construct_map_or_apply(setattr_lambda, federated_comp)


def construct_named_tuple_setattr_lambda(named_tuple_signature, name,
                                         value_comp):
  """Constructs a building block for replacing one attribute in a named tuple.

  Returns an instance of `computation_building_blocks.Lambda` which takes an
  argument of type `computation_types.NamedTupleType` and returns a
  `computation_building_blocks.Tuple` which contains all the same elements as
  the argument, except the attribute `name` now has value `value_comp`. The
  Lambda constructed is the analogue of Python's `setattr` for the concrete
  type `named_tuple_signature`.

  Args:
    named_tuple_signature: Instance of `computation_types.NamedTupleType`, the
      type of the argument to the constructed
      `computation_building_blocks.Lambda`.
    name: String name of the attribute in the `named_tuple_signature` to replace
      with `value_comp`. Must be present as a name in `named_tuple_signature;
      otherwise we will raise an `AttributeError`.
    value_comp: Instance of
      `computation_building_blocks.ComputationBuildingBlock`, the value to place
      as attribute `name` in the argument of the returned function.

  Returns:
    An instance of `computation_building_blocks.Block` of functional type
    representing setting attribute `name` to value `value_comp` in its argument
    of type `named_tuple_signature`.

  Raises:
    TypeError: If the types of the arguments don't match the assumptions above.
    AttributeError: If `name` is not present as a named element in
      `named_tuple_signature`
  """
  py_typecheck.check_type(named_tuple_signature,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(value_comp,
                          computation_building_blocks.ComputationBuildingBlock)
  value_comp_placeholder = computation_building_blocks.Reference(
      'value_comp_placeholder', value_comp.type_signature)
  lambda_arg = computation_building_blocks.Reference('lambda_arg',
                                                     named_tuple_signature)
  if name not in dir(named_tuple_signature):
    raise AttributeError(
        'There is no such attribute as \'{}\' in this federated tuple. '
        'TFF does not allow for assigning to a nonexistent attribute. '
        'If you want to assign to \'{}\', you must create a new named tuple '
        'containing this attribute.'.format(name, name))
  elements = []
  for idx, (key, element_type) in enumerate(
      anonymous_tuple.to_elements(named_tuple_signature)):
    if key == name:
      if not type_utils.is_assignable_from(element_type,
                                           value_comp.type_signature):
        raise TypeError(
            '`setattr` has attempted to set element {} of type {} with incompatible type {}'
            .format(key, element_type, value_comp.type_signature))
      elements.append((key, value_comp_placeholder))
    else:
      elements.append(
          (key, computation_building_blocks.Selection(lambda_arg, index=idx)))
  return_tuple = computation_building_blocks.Tuple(elements)
  lambda_to_return = computation_building_blocks.Lambda(lambda_arg.name,
                                                        named_tuple_signature,
                                                        return_tuple)
  enclosing_block = computation_building_blocks.Block(
      [(value_comp_placeholder.name, value_comp)], lambda_to_return)
  return enclosing_block


def construct_map_or_apply(fn, arg):
  """Injects intrinsic to allow application of `fn` to federated `arg`.

  Args:
    fn: An instance of `computation_building_blocks.ComputationBuildingBlock` of
      functional type to be wrapped with intrinsic in order to call on `arg`.
    arg: `computation_building_blocks.ComputationBuildingBlock` instance of
      federated type for which to construct intrinsic in order to call `fn` on
      `arg`. `member` of `type_signature` of `arg` must be assignable to
      `parameter` of `type_signature` of `fn`.

  Returns:
    Returns a `computation_building_blocks.Intrinsic` which can call
    `fn` on `arg`.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  type_utils.check_assignable_from(fn.type_signature.parameter,
                                   arg.type_signature.member)
  if arg.type_signature.placement == placement_literals.SERVER:
    result_type = computation_types.FederatedType(fn.type_signature.result,
                                                  arg.type_signature.placement,
                                                  arg.type_signature.all_equal)
    intrinsic_type = computation_types.FunctionType(
        [fn.type_signature, arg.type_signature], result_type)
    intrinsic = computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_APPLY.uri, intrinsic_type)
    tup = computation_building_blocks.Tuple((fn, arg))
    return computation_building_blocks.Call(intrinsic, tup)
  elif arg.type_signature.placement == placement_literals.CLIENTS:
    return create_federated_map(fn, arg)


def construct_federated_getattr_comp(comp, name):
  """Function to construct computation for `federated_apply` of `__getattr__`.

  Constructs a `computation_building_blocks.ComputationBuildingBlock`
  which selects `name` from its argument, of type `comp.type_signature.member`,
  an instance of `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      with type signature `computation_types.FederatedType` whose `member`
      attribute is of type `computation_types.NamedTupleType`.
    name: String name of attribute to grab.

  Returns:
    Instance of `computation_building_blocks.Lambda` which grabs attribute
      according to `name` of its argument.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(name, six.string_types)
  element_names = [
      x for x, _ in anonymous_tuple.to_elements(comp.type_signature.member)
  ]
  if name not in element_names:
    raise ValueError('The federated value {} has no element of name {}'.format(
        comp, name))
  apply_input = computation_building_blocks.Reference(
      'x', comp.type_signature.member)
  selected = computation_building_blocks.Selection(apply_input, name=name)
  apply_lambda = computation_building_blocks.Lambda('x',
                                                    apply_input.type_signature,
                                                    selected)
  return apply_lambda


def construct_federated_getitem_comp(comp, key):
  """Function to construct computation for `federated_apply` of `__getitem__`.

  Constructs a `computation_building_blocks.ComputationBuildingBlock`
  which selects `key` from its argument, of type `comp.type_signature.member`,
  of type `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      with type signature `computation_types.FederatedType` whose `member`
      attribute is of type `computation_types.NamedTupleType`.
    key: Instance of `int` or `slice`, key used to grab elements from the member
      of `comp`. implementation of slicing for `ValueImpl` objects with
      `type_signature` `computation_types.NamedTupleType`.

  Returns:
    Instance of `computation_building_blocks.Lambda` which grabs slice
      according to `key` of its argument.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(key, (int, slice))
  apply_input = computation_building_blocks.Reference(
      'x', comp.type_signature.member)
  if isinstance(key, int):
    selected = computation_building_blocks.Selection(apply_input, index=key)
  else:
    elems = anonymous_tuple.to_elements(comp.type_signature.member)
    index_range = six.moves.range(*key.indices(len(elems)))
    elem_list = []
    for k in index_range:
      elem_list.append(
          (elems[k][0],
           computation_building_blocks.Selection(apply_input, index=k)))
    selected = computation_building_blocks.Tuple(elem_list)
  apply_lambda = computation_building_blocks.Lambda('x',
                                                    apply_input.type_signature,
                                                    selected)
  return apply_lambda


def create_federated_map(fn, arg):
  r"""Creates a called federated map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A functional `computation_building_blocks.ComputationBuildingBlock` to
      use as the function.
    arg: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      argument.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  type_utils.check_federated_type(arg.type_signature)
  parameter_type = computation_types.FederatedType(fn.type_signature.parameter,
                                                   placement_literals.CLIENTS,
                                                   False)
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                placement_literals.CLIENTS,
                                                False)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, parameter_type), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP.uri, intrinsic_type)
  tup = computation_building_blocks.Tuple((fn, arg))
  return computation_building_blocks.Call(intrinsic, tup)


def construct_federated_zip_of_two_tuple(comp):
  """Zips a named tuple with two federated elements, dropping any names.

  It is necessary to drop names due to the type signature of federated zip:

                            <T@P,U@P>-><T,U>@P

  Tuples with named elements are not allowed in this type signature. This
  function, therefore, selects the 0th and 1st elements from its argument
  `comp`, ensuring that the resulting tuple is unnamed. It is this tuple that
  is zipped and returned.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      with type `computation_types.NamedTupleType` and whose elements are of
      type `computation_types.FederatedType`, all with the same placement.
      `comp` represents the building block we wish to zip.

  Returns:
    A zipped version of comp; an instance of
    `computation_building_blocks.ComputationBuildingBlock` of federated type,
    with member a named two-tuple with `None` for both names.

  Raises:
    TypeError: If the types don't match those described above, there are
    multiple distinct placements present in the argument tuple, or the
    placement on the argument tuple is currently unsupported.
    ValueError: If the argument tuple does not have two elements.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.NamedTupleType)
  zip_uris = {
      placement_literals.CLIENTS: intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri,
      placement_literals.SERVER: intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri,
  }
  zip_all_equal = {
      placement_literals.CLIENTS: False,
      placement_literals.SERVER: True,
  }
  output_placement = comp.type_signature[0].placement
  if output_placement not in zip_uris:
    raise TypeError('The argument must have components placed at SERVER or '
                    'CLIENTS')
  output_all_equal_bit = zip_all_equal[output_placement]
  for _, elem in anonymous_tuple.to_elements(comp.type_signature):
    py_typecheck.check_type(elem, computation_types.FederatedType)
    if elem.placement != output_placement:
      raise TypeError(
          'The argument to `construct_unnamed_federated_zip` must be a tuple of '
          'federated types, all of the same placement; you have passed in a '
          'value of placement {}, which conflicts with placement {}.'.format(
              elem.placement, output_placement))
  num_elements = len(anonymous_tuple.to_elements(comp.type_signature))
  if num_elements != 2:
    raise ValueError('The argument of zip_two_tuple must be a 2-tuple, '
                     'not an {}-tuple'.format(num_elements))
  result_type = computation_types.FederatedType(
      [e.member for _, e in anonymous_tuple.to_elements(comp.type_signature)],
      output_placement, output_all_equal_bit)

  def _adjust_all_equal_bit(x):
    return computation_types.FederatedType(x.member, x.placement,
                                           output_all_equal_bit)

  adjusted_input_type = computation_types.NamedTupleType([
      _adjust_all_equal_bit(v)
      for _, v in anonymous_tuple.to_elements(comp.type_signature)
  ])
  selected_input = computation_building_blocks.Tuple([
      computation_building_blocks.Selection(comp, index=0),
      computation_building_blocks.Selection(comp, index=1)
  ])
  concretized_intrinsic = computation_building_blocks.Intrinsic(
      zip_uris[output_placement],
      computation_types.FunctionType(adjusted_input_type, result_type))
  zipped = computation_building_blocks.Call(concretized_intrinsic,
                                            selected_input)
  return zipped


def construct_naming_function(tuple_type_to_name, names_to_add):
  """Constructs a function which names tuple elements via `names_to_add`.

  Certain intrinsics, e.g. `federated_zip`, only accept unnamed tuples as
  arguments, and can only produce unnamed tuples as their outputs. This is not
  necessarily desirable behavior, as it necessitates dropping any names that
  exist before the zip. This function is intended to provide a remedy for this
  shortcoming, so that a tuple can be renamed after it is passed through the
  `federated_zip` intrinsic.

  Args:
    tuple_type_to_name: Instance of `computation_types.NamedTupleType`, the type
      to populate with names from `names_to_add`.
    names_to_add: Python `tuple` or `list` containing instances of type `str` or
      `None`, the names to give to `tuple_type_to_name`.

  Returns:
    An instance of `computation_building_blocks.Lambda` representing a function
    which takes an argument of type `tuple_type_to_name` and returns the same
    argument, but with the names from `names_to_add` attached to the type
    signature.

  Raises:
    TypeError: If the types do not match the description above.
    ValueError: If `tuple_type_to_name` and `names_to_add` are of different
    lengths.
  """
  py_typecheck.check_type(tuple_type_to_name, computation_types.NamedTupleType)
  py_typecheck.check_type(names_to_add, (list, tuple))
  element_types_to_accept = six.string_types + (type(None),)
  if not all(isinstance(x, element_types_to_accept) for x in names_to_add):
    raise TypeError('`names_to_add` must contain only instances of `str` or '
                    'NoneType; you have passed in {}'.format(names_to_add))

  if len(names_to_add) != len(tuple_type_to_name):
    raise ValueError(
        'Number of elements in `names_to_add` must match number of element in '
        'the named tuple type `tuple_type_to_name`; here, `names_to_add` has '
        '{} elements and `tuple_type_to_name` has {}.'.format(
            len(names_to_add), len(tuple_type_to_name)))
  naming_lambda_arg = computation_building_blocks.Reference(
      'x', tuple_type_to_name)

  def _create_tuple_element(i):
    return (names_to_add[i],
            computation_building_blocks.Selection(naming_lambda_arg, index=i))

  named_result = computation_building_blocks.Tuple(
      [_create_tuple_element(k) for k in range(len(names_to_add))])
  return computation_building_blocks.Lambda('x',
                                            naming_lambda_arg.type_signature,
                                            named_result)
