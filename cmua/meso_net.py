import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import h5py # For loading TensorFlow's .h5 files

# Implementation of the Meso4 model
class Meso4(nn.Module):
    """
    PyTorch implementation of the Meso4 model from MesoNet
    Output dimension: 16 (in feature extraction mode)
    """
    def __init__(self):
        super(Meso4, self).__init__()

        # First Conv block
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second Conv block
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third Conv block
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth Conv block
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(4, 4)

        # Dense layers
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        # 256x256 -> 8x8 (after several pooling layers)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Classification
        x = self.flatten(x)
        x = self.dropout1(x)
        features = self.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

    def extract_features(self, x):
        """
        Performs only feature extraction (excluding classification layers)
        return: 16-dimensional feature vector
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc1(x))

        return x  # 16-dimensional feature vector

# Implementation of the Inception layer for MesoInception4
class InceptionLayer(nn.Module):
    def __init__(self, in_channels, a, b, c, d):
        super(InceptionLayer, self).__init__()

        # Branch 1: 1x1 conv
        self.branch1 = nn.Conv2d(in_channels, a, kernel_size=1)

        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch2_1 = nn.Conv2d(in_channels, b, kernel_size=1)
        self.branch2_2 = nn.Conv2d(b, b, kernel_size=3, padding=1)

        # Branch 3: 1x1 conv -> dilated 3x3 conv (rate=2)
        self.branch3_1 = nn.Conv2d(in_channels, c, kernel_size=1)
        self.branch3_2 = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2)

        # Branch 4: 1x1 conv -> dilated 3x3 conv (rate=3)
        self.branch4_1 = nn.Conv2d(in_channels, d, kernel_size=1)
        self.branch4_2 = nn.Conv2d(d, d, kernel_size=3, padding=3, dilation=3)

    def forward(self, x):
        branch1 = F.relu(self.branch1(x))

        branch2 = F.relu(self.branch2_1(x))
        branch2 = F.relu(self.branch2_2(branch2))

        branch3 = F.relu(self.branch3_1(x))
        branch3 = F.relu(self.branch3_2(branch3))

        branch4 = F.relu(self.branch4_1(x))
        branch4 = F.relu(self.branch4_2(branch4))

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# Implementation of the MesoInception4 model
class MesoInception4(nn.Module):
    """
    PyTorch implementation of the MesoInception4 model from MesoNet
    Output dimension: 16 (in feature extraction mode)
    """
    def __init__(self):
        super(MesoInception4, self).__init__()

        # First Inception Layer
        self.inception1 = InceptionLayer(3, 1, 4, 4, 2)
        self.bn1 = nn.BatchNorm2d(11)  # 1+4+4+2
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second Inception Layer
        self.inception2 = InceptionLayer(11, 2, 4, 4, 2)
        self.bn2 = nn.BatchNorm2d(12)  # 2+4+4+2
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third Conv Layer
        self.conv3 = nn.Conv2d(12, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth Conv Layer
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(4, 4)

        # Dense Layers
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)  # 256x256 -> 8x8
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.inception1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.inception2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.dropout1(x)
        features = self.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

    def extract_features(self, x):
        """
        Performs only feature extraction (excluding classification layers)
        return: 16-dimensional feature vector
        """
        x = self.inception1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.inception2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc1(x))

        return x  # 16-dimensional feature vector


def convert_tf_weights_to_pytorch(h5_path, pytorch_model, model_type):
    """
    Function to load TensorFlow .h5 weights into a PyTorch model

    Args:
        h5_path: Path to the .h5 weights file
        pytorch_model: The PyTorch model to load the weights into
        model_type: 'meso4' or 'mesoinception4'
    """
    if not os.path.exists(h5_path):
        print(f"Weight file not found: {h5_path}")
        return

    # Load the h5 file
    h5_file = h5py.File(h5_path, 'r')

    if model_type == 'meso4':
        layer_mapping = {
            'conv2d_5': ['conv1.weight', 'conv1.bias'], # Modify layer_mapping to match the Meso4 model weights file structure
            'batch_normalization_5': ['bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var'],
            'conv2d_6': ['conv2.weight', 'conv2.bias'],
            'batch_normalization_6': ['bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var'],
            'conv2d_7': ['conv3.weight', 'conv3.bias'],
            'batch_normalization_7': ['bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var'],
            'conv2d_8': ['conv4.weight', 'conv4.bias'],
            'batch_normalization_8': ['bn4.weight', 'bn4.bias', 'bn4.running_mean', 'bn4.running_var'],
            'dense_3': ['fc1.weight', 'fc1.bias'],
            'dense_4': ['fc2.weight', 'fc2.bias']
        }
    elif model_type == 'mesoinception4':
        layer_mapping = {
            'conv2d_1': ['inception1.branch1.weight', 'inception1.branch1.bias'], # inception1 branch1 (1x1 conv)
            'conv2d_2': ['inception1.branch2_1.weight', 'inception1.branch2_1.bias'], # inception1 branch2_1 (1x1 conv)
            'conv2d_3': ['inception1.branch2_2.weight', 'inception1.branch2_2.bias'], # inception1 branch2_2 (3x3 conv)
            'conv2d_4': ['inception1.branch3_1.weight', 'inception1.branch3_1.bias'], # inception1 branch3_1 (1x1 conv)
            'conv2d_5': ['inception1.branch3_2.weight', 'inception1.branch3_2.bias'], # inception1 branch3_2 (dilated 3x3 conv)
            'conv2d_6': ['inception1.branch4_1.weight', 'inception1.branch4_1.bias'], # inception1 branch4_1 (1x1 conv)
            'conv2d_7': ['inception1.branch4_2.weight', 'inception1.branch4_2.bias'], # inception1 branch4_2 (dilated 3x3 conv)
            'batch_normalization_1': ['bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var'], # BN after inception1

            'conv2d_8': ['inception2.branch1.weight', 'inception2.branch1.bias'], # inception2 branch1 (1x1 conv)
            'conv2d_9': ['inception2.branch2_1.weight', 'inception2.branch2_1.bias'], # inception2 branch2_1 (1x1 conv)
            'conv2d_10': ['inception2.branch2_2.weight', 'inception2.branch2_2.bias'], # inception2 branch2_2 (3x3 conv)
            'conv2d_11': ['inception2.branch3_1.weight', 'inception2.branch3_1.bias'], # inception2 branch3_1 (1x1 conv)
            'conv2d_12': ['inception2.branch3_2.weight', 'inception2.branch3_2.bias'], # inception2 branch3_2 (dilated 3x3 conv)
            'conv2d_13': ['inception2.branch4_1.weight', 'inception2.branch4_1.bias'], # inception2 branch4_1 (1x1 conv)
            'conv2d_14': ['inception2.branch4_2.weight', 'inception2.branch4_2.bias'], # inception2 branch4_2 (dilated 3x3 conv)
            'batch_normalization_2': ['bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var'], # BN after inception2

            'conv2d_15': ['conv3.weight', 'conv3.bias'], # conv3 layer
            'batch_normalization_3': ['bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var'], # bn3 layer
            'conv2d_16': ['conv4.weight', 'conv4.bias'], # conv4 layer
            'batch_normalization_4': ['bn4.weight', 'bn4.bias', 'bn4.running_mean', 'bn4.running_var'], # bn4 layer

            'dense_1': ['fc1.weight', 'fc1.bias'], # fc1 layer
            'dense_2': ['fc2.weight', 'fc2.bias']  # fc2 layer
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of 'meso4' or 'mesoinception4'.")

    state_dict = pytorch_model.state_dict()

    for tf_layer_name, pt_params in layer_mapping.items():
        if tf_layer_name in h5_file:
            tf_layer = h5_file[tf_layer_name]
            for i, param_name in enumerate(pt_params):
                try:
                    inner_tf_layer = tf_layer[tf_layer_name] # Access the inner group
                    dataset_name = None
                    if ('weight' in param_name and 'conv' in param_name) or ('weight' in param_name and 'branch' in param_name):
                        dataset_name = 'kernel:0'
                    elif ('bias' in param_name and 'conv' in param_name) or ('bias' in param_name and 'branch' in param_name):
                        dataset_name = 'bias:0'
                    elif 'weight' in param_name and 'fc' in param_name:
                        dataset_name = 'kernel:0'
                    elif 'bias' in param_name and 'fc' in param_name:
                        dataset_name = 'bias:0'
                    elif 'bn' in param_name and 'weight' in param_name:
                        dataset_name = 'gamma:0' # BN gamma is weight in PyTorch
                    elif 'bn' in param_name and 'bias' in param_name:
                        dataset_name = 'beta:0'  # BN beta is bias in PyTorch
                    elif 'bn' in param_name and 'running_mean' in param_name:
                        dataset_name = 'moving_mean:0'
                    elif 'bn' in param_name and 'running_var' in param_name:
                        dataset_name = 'moving_variance:0'

                    if dataset_name:
                        tf_weight = inner_tf_layer[dataset_name] # Access dataset by name from inner group
                        pt_weight = state_dict[param_name]

                        # Convolutional layer weights
                        if ('conv' in param_name and 'weight' in param_name) or ('branch' in param_name and 'weight' in param_name):
                            # TF: [height, width, in_channels, out_channels] -> PyTorch: [out_channels, in_channels, height, width]
                            tf_weight = np.transpose(tf_weight, (3, 2, 0, 1))
                        # Dense layer weights
                        elif 'fc' in param_name and 'weight' in param_name:
                            # TF: [in_features, out_features] -> PyTorch: [out_features, in_features]
                            tf_weight = np.transpose(tf_weight)
                        # Batch Norm running mean and var
                        elif 'bn' in param_name and ('running_mean' in param_name or 'running_var' in param_name):
                            tf_weight = np.array(tf_weight) # convert h5py dataset to numpy array
                        else:
                            tf_weight = np.array(tf_weight) # convert h5py dataset to numpy array

                        assert pt_weight.shape == tf_weight.shape, f"Shape mismatch for {param_name}: PyTorch shape {pt_weight.shape}, TF shape {tf_weight.shape}"
                        state_dict[param_name].copy_(torch.from_numpy(tf_weight))

                except Exception as e:
                    print(f"[Debug] Weight loading failed! -> {e}")

    pytorch_model.load_state_dict(state_dict)
    print(f"Successfully loaded weights into {model_type} model.")


if __name__ == "__main__":
    # Create model instances
    meso4_model = Meso4()
    meso_inception4_model = MesoInception4()

    # Weight file paths
    meso4_h5_path_df = "./weights/Meso4_DF.h5"  # Meso4 weights for Deepfakes dataset
    meso4_h5_path_f2f = "./weights/Meso4_F2F.h5" # Meso4 weights for Face2Face dataset
    meso_inception4_h5_path_df = "./weights/MesoInception_DF.h5" # MesoInception4 weights for Deepfakes dataset
    meso_inception4_h5_path_f2f = "./weights/MesoInception_F2F.h5" # MesoInception4 weights for Face2Face dataset


    # Load weights and test
    try:
        convert_tf_weights_to_pytorch(meso4_h5_path_df, meso4_model, 'meso4')
        print("Meso4 (Deepfakes) weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load Meso4 (Deepfakes) weights: {e}")

    try:
        convert_tf_weights_to_pytorch(meso4_h5_path_f2f, meso4_model, 'meso4')
        print("Meso4 (Face2Face) weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load Meso4 (Face2Face) weights: {e}")

    try:
        convert_tf_weights_to_pytorch(meso_inception4_h5_path_df, meso_inception4_model, 'mesoinception4')
        print("MesoInception4 (Deepfakes) weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load MesoInception4 (Deepfakes) weights: {e}")

    try:
        convert_tf_weights_to_pytorch(meso_inception4_h5_path_f2f, meso_inception4_model, 'mesoinception4')
        print("MesoInception4 (Face2Face) weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load MesoInception4 (Face2Face) weights: {e}")


    # Feature extraction test
    meso4_model.eval()
    meso_inception4_model.eval()

    dummy_input = torch.randn(1, 3, 256, 256) # Batch size 1, RGB channels, image size 256x256

    with torch.no_grad():
        meso4_features = meso4_model.extract_features(dummy_input)
        meso_inception4_features = meso_inception4_model.extract_features(dummy_input)

    print("\nMeso4 feature vector shape:", meso4_features.shape) # Expected: torch.Size([1, 16])
    print("MesoInception4 feature vector shape:", meso_inception4_features.shape) # Expected: torch.Size([1, 16])