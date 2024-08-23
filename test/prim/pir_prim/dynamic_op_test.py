# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import unittest
from datetime import datetime
from inspect import Parameter, Signature

import numpy as np

import paddle
from paddle.base import core
from paddle.static import InputSpec


def create_function(func, *args, **kwargs):
    def new_func(*args):
        return func(*args, **kwargs)

    new_sig = Signature(
        [Parameter(item, Parameter.POSITIONAL_ONLY) for item in args]
    )
    new_func.__signature__ = new_sig
    return new_func


def get_arg_names(func):
    return [param.name for param in inspect.signature(func).parameters.values()]


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


class DynamicOpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(datetime.now().year)
        cls.check_cinn = False

    def wrap_to_list(self, x):
        if isinstance(x, tuple or list):
            return list(x)
        else:
            return [x]

    def get_dynamic_forward(self, args):
        return self.wrap_to_list(self.net(*args))

    def get_dynamic_backward(self, args):
        res = self.wrap_to_list(self.net(*args))
        for i in range(len(res)):
            res[i].backward()
        input_grads = [arg.gradient() for arg in args]
        return [res, input_grads]

    def get_static_foward(self, args, input_spec):
        core._set_prim_all_enabled(True)
        static_func = apply_to_static(
            self.net, use_cinn=self.enable_cinn, input_spec=input_spec
        )
        static_func.eval()
        res = static_func(*args)
        core._set_prim_all_enabled(False)
        return self.wrap_to_list(res)

    def get_static_backward(self, args, input_spec):
        core._set_prim_all_enabled(True)
        static_func = apply_to_static(
            self.net, use_cinn=self.enable_cinn, input_spec=input_spec
        )
        static_func.train()
        res = self.wrap_to_list(static_func(*args))
        for i in range(len(res)):
            res[i].backward()
        input_grads = [arg.gradient() for arg in args]
        core._set_prim_all_enabled(False)
        return [res, input_grads]

    def check_all_close_forward(self, references, actuals):
        for reference, actual in zip(references, actuals):
            for ref, act in zip(reference, actual):
                np.testing.assert_allclose(
                    ref, act, rtol=self.rtol, atol=self.atol
                )

    def check_all_close_backward(self, references, actuals):
        for reference, actual in zip(references[0], actuals[0]):
            for ref, act in zip(reference, actual):
                np.testing.assert_allclose(
                    ref, act, rtol=self.rtol, atol=self.atol
                )
        for ref_grad, act_grad in zip(references[1], actuals[1]):
            np.testing.assert_allclose(
                ref_grad, act_grad, rtol=self.rtol, atol=self.atol
            )

    def prepare_args_input_spec(self, stop_gradient=True):
        # find formals
        formals = get_arg_names(self.net)
        # get input args and input_spec of `apply_to_static`
        args = []
        input_spec = []
        for name in formals:
            args.append(
                paddle.to_tensor(
                    getattr(self, name), stop_gradient=stop_gradient
                )
            )
            item = self.inputs[name]
            # assemble input spec
            input_spec.append(
                InputSpec(shape=item['dynamic_shape'], dtype=item['dtype'])
            )

        return args, input_spec

    def check_output(self):
        # forward
        args, input_spec = self.prepare_args_input_spec()
        references = self.get_dynamic_forward(args)
        actuals = self.get_static_foward(args, input_spec)
        self.check_all_close_forward(references, actuals)

    def check_grad(self):
        args_0, _ = self.prepare_args_input_spec(stop_gradient=False)
        args_1, input_spec = self.prepare_args_input_spec(stop_gradient=False)
        reference = self.get_dynamic_backward(args_0)
        actual = self.get_static_backward(args_1, input_spec)
        self.check_all_close_backward(reference, actual)
