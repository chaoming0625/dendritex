# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
# ==============================================================================

import numpy as np
loaded_params = np.load('golgi_morphology.npz')

connection = loaded_params['connection']
L = loaded_params['L']               # um
diam = loaded_params['diam']         # um
Ra = loaded_params['Ra']             # ohm * cm
cm = loaded_params['cm']             # uF / cm ** 2


index_soma = loaded_params['index_soma']
index_axon = loaded_params['index_axon']
index_dend_basal = loaded_params['index_dend_basal']
index_dend_apical = loaded_params['index_dend_apical']
