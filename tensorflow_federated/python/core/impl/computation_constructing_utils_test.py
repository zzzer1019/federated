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
"""Tests for computation_constructing_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


class ComputationConstructionUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getitem_comp_construction(self, placement):
    federated_value = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement, True))
    get_0_comp = computation_constructing_utils.construct_federated_getitem_comp(
        federated_value, 0)
    self.assertEqual(str(get_0_comp), '(x -> x[0])')
    get_slice_comp = computation_constructing_utils.construct_federated_getitem_comp(
        federated_value, slice(None, None, -1))
    self.assertEqual(str(get_slice_comp), '(x -> <b=x[1],a=x[0]>)')

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getattr_comp_construction(self, placement):
    federated_value = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement, True))
    get_a_comp = computation_constructing_utils.construct_federated_getattr_comp(
        federated_value, 'a')
    self.assertEqual(str(get_a_comp), '(x -> x.a)')
    get_b_comp = computation_constructing_utils.construct_federated_getattr_comp(
        federated_value, 'b')
    self.assertEqual(str(get_b_comp), '(x -> x.b)')
    non_federated_arg = computation_building_blocks.Reference(
        'test',
        computation_types.NamedTupleType([('a', tf.int32), ('b', tf.bool)]))
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.construct_federated_getattr_comp(
          non_federated_arg, 'a')
    with self.assertRaisesRegex(ValueError, 'has no element of name c'):
      _ = computation_constructing_utils.construct_federated_getattr_comp(
          federated_value, 'c')

  def test_intrinsic_construction_server(self):
    federated_comp = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement_literals.SERVER, True))
    arg_ref = computation_building_blocks.Reference('x', [('a', tf.int32),
                                                          ('b', tf.bool)])
    return_val = computation_building_blocks.Selection(arg_ref, name='a')
    non_federated_fn = computation_building_blocks.Lambda(
        'x', arg_ref.type_signature, return_val)
    intrinsic = computation_constructing_utils.construct_map_or_apply(
        non_federated_fn, federated_comp)
    self.assertEqual(str(intrinsic), 'federated_apply(<(x -> x.a),test>)')

  def test_intrinsic_construction_clients(self):
    federated_comp = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement_literals.CLIENTS, True))
    arg_ref = computation_building_blocks.Reference('x', [('a', tf.int32),
                                                          ('b', tf.bool)])
    return_val = computation_building_blocks.Selection(arg_ref, name='a')
    non_federated_fn = computation_building_blocks.Lambda(
        'x', arg_ref.type_signature, return_val)
    intrinsic = computation_constructing_utils.construct_map_or_apply(
        non_federated_fn, federated_comp)
    self.assertEqual(intrinsic.tff_repr, 'federated_map(<(x -> x.a),test>)')
    self.assertEqual(str(intrinsic.type_signature), '{int32}@CLIENTS')

  def test_intrinsic_construction_fails_bad_type(self):
    x = computation_building_blocks.Reference('x', tf.int32)
    bad_lambda = computation_building_blocks.Lambda('x', tf.int32, x)
    x = computation_building_blocks.Reference('x', tf.int32)
    clients_ref = computation_building_blocks.Reference(
        'y',
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_map_or_apply(
          bad_lambda, clients_ref)

  def test_federated_getitem_call_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([tf.int32]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_getitem_call(
          value_impl.to_value(x), 0)

  def test_federated_getattr_call_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([('x', tf.int32)]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_getattr_call(
          value_impl.to_value(x), 'x')

  def test_federated_getitem_comp_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([tf.int32]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_getitem_comp(
          value_impl.to_value(x), 0)

  def test_federated_getattr_comp_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([('x', tf.int32)]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_getattr_comp(
          value_impl.to_value(x), 'x')

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getitem_call_named(self, placement):
    federated_comp_named = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement, True))
    self.assertEqual(
        str(federated_comp_named.type_signature.member), '<a=int32,b=bool>')
    idx_0 = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_named, 0)
    idx_1 = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_named, 1)
    self.assertIsInstance(idx_0.type_signature, computation_types.FederatedType)
    self.assertIsInstance(idx_1.type_signature, computation_types.FederatedType)
    self.assertEqual(str(idx_0.type_signature.member), 'int32')
    self.assertEqual(str(idx_1.type_signature.member), 'bool')
    type_utils.check_federated_value_placement(
        value_impl.to_value(idx_0, None, context_stack_impl.context_stack),
        placement)
    type_utils.check_federated_value_placement(
        value_impl.to_value(idx_1, None, context_stack_impl.context_stack),
        placement)
    flipped = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_named, slice(None, None, -1))
    self.assertIsInstance(flipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(flipped.type_signature.member), '<b=bool,a=int32>')
    type_utils.check_federated_value_placement(
        value_impl.to_value(flipped, None, context_stack_impl.context_stack),
        placement)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getitem_call_unnamed(self, placement):
    federated_comp_unnamed = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([tf.int32, tf.bool], placement, True))
    self.assertEqual(
        str(federated_comp_unnamed.type_signature.member), '<int32,bool>')
    unnamed_idx_0 = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_unnamed, 0)
    unnamed_idx_1 = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_unnamed, 1)
    self.assertIsInstance(unnamed_idx_0.type_signature,
                          computation_types.FederatedType)
    self.assertIsInstance(unnamed_idx_1.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(unnamed_idx_0.type_signature.member), 'int32')
    self.assertEqual(str(unnamed_idx_1.type_signature.member), 'bool')
    type_utils.check_federated_value_placement(
        value_impl.to_value(unnamed_idx_0, None,
                            context_stack_impl.context_stack), placement)
    type_utils.check_federated_value_placement(
        value_impl.to_value(unnamed_idx_1, None,
                            context_stack_impl.context_stack), placement)
    unnamed_flipped = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_unnamed, slice(None, None, -1))
    self.assertIsInstance(unnamed_flipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(unnamed_flipped.type_signature.member), '<bool,int32>')
    type_utils.check_federated_value_placement(
        value_impl.to_value(unnamed_flipped, None,
                            context_stack_impl.context_stack), placement)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getattr_call_named(self, placement):
    federated_comp_named = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32),
                                         ('b', tf.bool), tf.int32], placement,
                                        True))
    self.assertEqual(
        str(federated_comp_named.type_signature.member),
        '<a=int32,b=bool,int32>')
    name_a = computation_constructing_utils.construct_federated_getattr_call(
        federated_comp_named, 'a')
    name_b = computation_constructing_utils.construct_federated_getattr_call(
        federated_comp_named, 'b')
    self.assertIsInstance(name_a.type_signature,
                          computation_types.FederatedType)
    self.assertIsInstance(name_b.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(name_a.type_signature.member), 'int32')
    self.assertEqual(str(name_b.type_signature.member), 'bool')
    type_utils.check_federated_value_placement(
        value_impl.to_value(name_a, None, context_stack_impl.context_stack),
        placement)
    type_utils.check_federated_value_placement(
        value_impl.to_value(name_b, None, context_stack_impl.context_stack),
        placement)
    with self.assertRaisesRegex(ValueError, 'has no element of name c'):
      _ = computation_constructing_utils.construct_federated_getattr_call(
          federated_comp_named, 'c')

  def test_construct_setattr_named_tuple_type_fails_on_bad_type(self):
    bad_type = computation_types.FederatedType([('a', tf.int32)],
                                               placement_literals.CLIENTS)
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.construct_named_tuple_setattr_lambda(
          bad_type, 'a', value_comp)

  def test_construct_setattr_named_tuple_type_fails_on_none_name(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.construct_named_tuple_setattr_lambda(
          good_type, None, value_comp)

  def test_construct_setattr_named_tuple_type_fails_on_none_value(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32)])
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.construct_named_tuple_setattr_lambda(
          good_type, 'a', None)

  def test_construct_setattr_named_tuple_type_fails_implicit_type_conversion(
      self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaisesRegex(TypeError, 'incompatible type'):
      _ = computation_constructing_utils.construct_named_tuple_setattr_lambda(
          good_type, 'b', value_comp)

  def test_construct_setattr_named_tuple_type_fails_unknown_name(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(AttributeError):
      _ = computation_constructing_utils.construct_named_tuple_setattr_lambda(
          good_type, 'c', value_comp)

  def test_construct_setattr_named_tuple_type_replaces_single_element(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    lam = computation_constructing_utils.construct_named_tuple_setattr_lambda(
        good_type, 'a', value_comp)
    self.assertEqual(
        lam.tff_repr,
        '(let value_comp_placeholder=x in (lambda_arg -> <a=value_comp_placeholder,b=lambda_arg[1]>))'
    )

  def test_construct_setattr_named_tuple_type_skips_unnamed_element(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  (None, tf.float32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    lam = computation_constructing_utils.construct_named_tuple_setattr_lambda(
        good_type, 'a', value_comp)
    self.assertEqual(
        lam.tff_repr,
        '(let value_comp_placeholder=x in (lambda_arg -> <a=value_comp_placeholder,lambda_arg[1],b=lambda_arg[2]>))'
    )

  def test_construct_setattr_named_tuple_type_leaves_type_signature_unchanged(
      self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  (None, tf.float32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    lam = computation_constructing_utils.construct_named_tuple_setattr_lambda(
        good_type, 'a', value_comp)
    self.assertTrue(
        type_utils.are_equivalent_types(lam.type_signature.parameter,
                                        lam.type_signature.result))

  def test_federated_setattr_call_fails_on_none_federated_comp(self):
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.construct_federated_setattr_call(
          None, 'a', value_comp)

  def test_federated_setattr_call_fails_non_federated_type(self):
    bad_type = computation_types.NamedTupleType([('a', tf.int32),
                                                 (None, tf.float32),
                                                 ('b', tf.bool)])
    bad_comp = computation_building_blocks.Data('data', bad_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.construct_federated_setattr_call(
          bad_comp, 'a', value_comp)

  def test_federated_setattr_call_fails_on_none_name(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    acceptable_comp = computation_building_blocks.Data('data', good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.construct_federated_setattr_call(
          acceptable_comp, None, value_comp)

  def test_federated_setattr_call_fails_on_none_value(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    acceptable_comp = computation_building_blocks.Data('data', good_type)

    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.construct_federated_setattr_call(
          acceptable_comp, 'a', None)

  def test_federated_setattr_call_constructs_correct_intrinsic_clients(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.construct_federated_setattr_call(
        federated_comp, 'a', value_comp)
    self.assertEqual(federated_setattr.function.uri,
                     intrinsic_defs.FEDERATED_MAP.uri)

  def test_federated_setattr_call_constructs_correct_intrinsic_server(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.SERVER)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.construct_federated_setattr_call(
        federated_comp, 'a', value_comp)
    self.assertEqual(federated_setattr.function.uri,
                     intrinsic_defs.FEDERATED_APPLY.uri)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_federated_setattr_call_leaves_type_signatures_alone(self, placement):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type, placement)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.construct_federated_setattr_call(
        federated_comp, 'a', value_comp)
    self.assertTrue(
        type_utils.are_equivalent_types(federated_setattr.type_signature,
                                        federated_comp.type_signature))

  def test_federated_setattr_call_constructs_correct_computation_clients(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.construct_federated_setattr_call(
        federated_comp, 'a', value_comp)
    self.assertEqual(
        federated_setattr.tff_repr,
        'federated_map(<(let value_comp_placeholder=x in (lambda_arg -> <a=value_comp_placeholder,lambda_arg[1],b=lambda_arg[2]>)),federated_comp>)'
    )

  def test_federated_setattr_call_constructs_correct_computation_server(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.SERVER)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.construct_federated_setattr_call(
        federated_comp, 'a', value_comp)
    self.assertEqual(
        federated_setattr.tff_repr,
        'federated_apply(<(let value_comp_placeholder=x in (lambda_arg -> <a=value_comp_placeholder,lambda_arg[1],b=lambda_arg[2]>)),federated_comp>)'
    )


class CreateFederatedMapTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = computation_building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map(None, arg)

  def test_raises_type_error_with_nonfunctional_fn(self):
    fn = computation_building_blocks.Reference('x', tf.int32)
    arg_type = computation_types.FederatedType(tf.bool, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map(fn, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map(fn, None)

  def test_raises_type_error_with_nonfederated_arg(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg = computation_building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map(fn, arg)

  def test_returns_federated_map(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    comp = computation_constructing_utils.create_federated_map(fn, arg)
    self.assertEqual(comp.tff_repr, 'federated_map(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), '{int32}@CLIENTS')


class FederatedZipTwoTupleConstructionTest(absltest.TestCase):

  def test_construct_federated_zip_of_two_tuple_raises_on_none(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_zip_of_two_tuple(None)

  def test_construct_federated_zip_of_two_tuple_raises_wrong_type(self):
    data = computation_building_blocks.Data('data',
                                            computation_types.to_type(tf.int32))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_zip_of_two_tuple(data)

  def test_construct_federated_zip_of_two_tuple_raises_bad_placement(self):
    bad_placement = placement_literals.PlacementLiteral('mock', 'mock', False,
                                                        'mock')
    federated_type = computation_types.FederatedType(tf.int32, bad_placement)
    data = computation_building_blocks.Data('data', federated_type)
    tup = computation_building_blocks.Tuple([data, data])
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_zip_of_two_tuple(tup)

  def test_construct_federated_zip_of_two_tuple_raises_conflicting_placements(
      self):
    federated_at_server = computation_types.FederatedType(
        tf.int32, placement_literals.SERVER)
    federated_at_clients = computation_types.FederatedType(
        tf.int32, placement_literals.CLIENTS)
    data = computation_building_blocks.Data(
        'data', [federated_at_server, federated_at_clients])
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_zip_of_two_tuple(data)

  def test_construct_federated_zip_of_two_tuple_raises_wrong_length(self):
    federated_at_clients = computation_types.FederatedType(
        tf.int32, placement_literals.CLIENTS)
    data = computation_building_blocks.Data('data', [federated_at_clients] * 3)
    with self.assertRaises(ValueError):
      computation_constructing_utils.construct_federated_zip_of_two_tuple(data)

  def test_construct_federated_zip_of_two_tuple_constructs_zip_of_correct_type_unnamed_tuple(
      self):
    federated_int = computation_types.FederatedType(tf.int32,
                                                    placement_literals.CLIENTS)
    federated_float = computation_types.FederatedType(
        tf.float32, placement_literals.CLIENTS)
    data = computation_building_blocks.Data('data',
                                            [federated_int, federated_float])
    zipped_data = computation_constructing_utils.construct_federated_zip_of_two_tuple(
        data)
    expected_zipped_type = computation_types.FederatedType(
        [tf.int32, tf.float32], placement_literals.CLIENTS)
    self.assertEqual(zipped_data.type_signature, expected_zipped_type)

  def test_construct_federated_zip_of_two_tuple_actually_drops_names(self):
    federated_int = computation_types.FederatedType(tf.int32,
                                                    placement_literals.CLIENTS)
    federated_float = computation_types.FederatedType(
        tf.float32, placement_literals.CLIENTS)
    data = computation_building_blocks.Data('data', [('a', federated_int),
                                                     ('b', federated_float)])
    zipped_data = computation_constructing_utils.construct_federated_zip_of_two_tuple(
        data)
    expected_zipped_type = computation_types.FederatedType(
        [tf.int32, tf.float32], placement_literals.CLIENTS)
    self.assertEqual(zipped_data.type_signature, expected_zipped_type)

  def test_lambda_to_drop_names_and_federated_zip_at_clients_repr(self):
    federated_int = computation_types.FederatedType(tf.int32,
                                                    placement_literals.CLIENTS)
    federated_float = computation_types.FederatedType(
        tf.float32, placement_literals.CLIENTS)
    arg = computation_building_blocks.Reference('arg', [('a', federated_int),
                                                        ('b', federated_float)])
    zipped = computation_constructing_utils.construct_federated_zip_of_two_tuple(
        arg)
    lam = computation_building_blocks.Lambda('arg', arg.type_signature, zipped)
    self.assertEqual(lam.tff_repr,
                     '(arg -> federated_zip_at_clients(<arg[0],arg[1]>))')

  def test_lambda_to_drop_names_and_federated_zip_at_server_repr(self):
    federated_int = computation_types.FederatedType(tf.int32,
                                                    placement_literals.SERVER)
    federated_float = computation_types.FederatedType(tf.float32,
                                                      placement_literals.SERVER)
    arg = computation_building_blocks.Reference('arg', [('a', federated_int),
                                                        ('b', federated_float)])
    zipped = computation_constructing_utils.construct_federated_zip_of_two_tuple(
        arg)
    lam = computation_building_blocks.Lambda('arg', arg.type_signature, zipped)
    self.assertEqual(lam.tff_repr,
                     '(arg -> federated_zip_at_server(<arg[0],arg[1]>))')


class ConstructNamingFunctionTest(absltest.TestCase):

  def test_construct_naming_function_raises_on_none(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_naming_function(None, ['a'])

  def test_construct_naming_function_raises_wrong_type(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_naming_function(
          computation_types.to_type(tf.int32), ['a'])

  def test_construct_naming_function_raises_on_naked_string(self):
    ntt = computation_types.NamedTupleType([tf.int32])
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_naming_function(ntt, 'a')

  def test_construct_naming_function_raises_list_of_ints(self):
    ntt = computation_types.NamedTupleType([tf.int32])
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_naming_function(ntt, [1])

  def test_construct_naming_function_raises_wrong_list_length(self):
    ntt = computation_types.NamedTupleType([tf.int32])
    with self.assertRaises(ValueError):
      computation_constructing_utils.construct_naming_function(ntt, ['a', 'b'])

  def test_construct_naming_function_constructs_function_of_correct_type_from_unnamed_tuple(
      self):
    ntt = computation_types.NamedTupleType([tf.int32, tf.float32])
    naming_fxn = computation_constructing_utils.construct_naming_function(
        ntt, ['a', 'b'])
    expected_function_type = computation_types.FunctionType(
        [tf.int32, tf.float32], [('a', tf.int32), ('b', tf.float32)])
    self.assertEqual(expected_function_type, naming_fxn.type_signature)

  def test_construct_naming_function_constructs_function_of_correct_type_from_named_tuple(
      self):
    ntt = computation_types.NamedTupleType([('c', tf.int32), ('d', tf.float32)])
    naming_fxn = computation_constructing_utils.construct_naming_function(
        ntt, ['a', 'b'])
    expected_function_type = computation_types.FunctionType([('c', tf.int32),
                                                             ('d', tf.float32)],
                                                            [('a', tf.int32),
                                                             ('b', tf.float32)])
    self.assertEqual(expected_function_type, naming_fxn.type_signature)

  def test_construct_naming_function_only_names_unnamed_tuple(self):
    ntt = computation_types.NamedTupleType([tf.int32, tf.float32])
    naming_fxn = computation_constructing_utils.construct_naming_function(
        ntt, ['a', 'b'])
    self.assertEqual(naming_fxn.tff_repr, '(x -> <a=x[0],b=x[1]>)')

  def test_construct_naming_function_only_overwrites_existing_names_in_tuple(
      self):
    ntt = computation_types.NamedTupleType([('c', tf.int32), ('d', tf.float32)])
    naming_fxn = computation_constructing_utils.construct_naming_function(
        ntt, ['a', 'b'])
    self.assertEqual(naming_fxn.tff_repr, '(x -> <a=x[0],b=x[1]>)')


if __name__ == '__main__':
  absltest.main()
