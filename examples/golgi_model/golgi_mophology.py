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
