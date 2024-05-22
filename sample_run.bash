# Training with MLP
## Backpropagation
python main_class_mlp.py --yaml './yaml/mlp/cifar10_bp.yaml'
# ## Counter-Current Learning
python main_class_mlp.py --yaml './yaml/mlp/cifar10_ccl.yaml'

# # Training with CNN
# ## Backpropagation
python main_class_cnn.py --yaml "./yaml/cnn/cifar10_bp.yaml"
# ## Counter-Current Learning
python main_class_cnn.py --yaml "./yaml/cnn/cifar10_ccl.yaml"

# # Auto-Encoder on STL-10
# ## Backpropagation
python main_ae.py --yaml './yaml/cnn_ae/stl10_bp_legacy.yaml'
# ## Counter-Current Learning
python main_ae.py --yaml './yaml/cnn_ae/stl10_ccl_legacy.yaml'