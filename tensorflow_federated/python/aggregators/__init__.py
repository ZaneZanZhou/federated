# Copyright 2020, The TensorFlow Federated Authors.
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
"""Libraries for constructing federated aggregation."""

from tensorflow_federated.python.aggregators.clipping_factory import ClippingFactory
from tensorflow_federated.python.aggregators.clipping_factory import ZeroingFactory
from tensorflow_federated.python.aggregators.dp_factory import DifferentiallyPrivateFactory
from tensorflow_federated.python.aggregators.encoded_factory import EncodedSumFactory
from tensorflow_federated.python.aggregators.factory import UnweightedAggregationFactory
from tensorflow_federated.python.aggregators.factory import WeightedAggregationFactory
from tensorflow_federated.python.aggregators.mean_factory import MeanFactory
from tensorflow_federated.python.aggregators.quantile_estimation import PrivateQuantileEstimationProcess
from tensorflow_federated.python.aggregators.secure_factory import SecureSumFactory
from tensorflow_federated.python.aggregators.sum_factory import SumFactory
