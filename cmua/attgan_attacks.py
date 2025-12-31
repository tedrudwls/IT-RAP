import torch
import torch.nn as nn

from torch.autograd import Variable as V

import torch_dct as dct

"""
Role: A class that integrally implements FGSM, I-FGSM, and PGD attacks.

Initialization:
epsilon: The maximum allowed noise size for the attack (based on L∞ norm).
a: The noise update size at each step.
"""
class AttackFunction(object):
    def __init__(self, config, model, device=None, epsilon=0.05, a=0.01):
        self.model = model # The Generator model passed from solver.py
        self.epsilon = epsilon # Maximum attack size (L∞ constraint)
        self.a = a # Step size (learning rate)
        self.loss_fn = nn.MSELoss().to(device) # Objective function (MSE between output and target)
        self.device = device # Computation device (CPU/GPU)

        self.config = config # Configuration passed from new_solver.py

        # Selects PGD or I-FGSM (whether to use random noise for PGD initialization)
        self.rand = True

        # Frequency mask member variables (initialized and reused)
        self.freq_mask_all = self.create_frequency_masks([1, 3, 256, 256], "ALL")
        self.freq_mask_low = self.create_frequency_masks([1, 3, 256, 256], "LOW")
        self.freq_mask_mid = self.create_frequency_masks([1, 3, 256, 256], "MID")
        self.freq_mask_high = self.create_frequency_masks([1, 3, 256, 256], "HIGH")

    """
    Role: Performs a basic I-FGSM attack.

    Operation:
    1. Uses the original image.
    2. Calculates gradients and updates the noise.
    3. At each step, limits the noise size to within epsilon and clips the image pixel values to [-1, 1].
    """
    def PGD(self, X_nat, y, c_trg):
        X = X_nat.clone().detach_()

        iter = self.config.pgd_iter

        for i in range(iter): # Iterate 'iter' times
            X.requires_grad = True

            # At each step, change "face attributes" with the Generator model using the image with added perturbation from that step.
            output = self.model(X, c_trg, mode='enc-dec') 

            self.model.zero_grad()
            
            # Calculate gradient
            # MSE loss with the target (y = image with face attributes changed by GAN from the original image)
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            # PGD update: move by step size (a=0.01) in the direction of the gradient.
            X_adv = X + self.a * grad.sign()

            # Limit the range of the perturbation.
            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon) 
            # After adding the perturbation, clip each pixel value of the image to always be within [-1, 1].
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_() 

        self.model.zero_grad()

        # The image with the optimized perturbation added, and the optimized perturbation itself.
        return X, X - X_nat


    def perturb_frequency_domain(self, X_nat, y, c_trg, freq_band='ALL', iter=1):
        """
        Function to perform FGSM attack in the frequency domain.
        
        Args:
            X_nat: Original image.
            y: Target label.
            c_trg: Target class.
            freq_band: Frequency band to attack ('LOW', 'MID', 'HIGH', 'ALL').
        
        Returns:
            Adversarial example, perturbation.
        """

        iter = self.config.dct_iter
        dct_coef = self.config.dct_coefficent
        dct_clamp = self.config.dct_clamp

        # Calculate DCT coefficients of the original image.
        X_nat_dct = torch.zeros_like(X_nat)
        for b in range(X_nat.shape[0]):
            for c in range(X_nat.shape[1]):
                X_nat_dct[b, c] = dct.dct_2d(X_nat[b, c])

        # Select the base mask for the requested frequency band.
        if freq_band == 'ALL':
            freq_mask = self.freq_mask_all
        elif freq_band == 'LOW':
            freq_mask = self.freq_mask_low
        elif freq_band == 'MID':
            freq_mask = self.freq_mask_mid
        elif freq_band == 'HIGH':
            freq_mask = self.freq_mask_high
        else:
            raise ValueError(f"Unsupported frequency band: {freq_band}")

        # Initialization: Perturbation to apply to DCT coefficients -> initialize to 0.
        eta_dct = torch.zeros_like(X_nat_dct)

        # Apply initial perturbation only to the selected frequency band.
        eta_dct = eta_dct * freq_mask

        for i in range(iter): # Iterate 'iter' times
            eta_dct.requires_grad = True

            # Current DCT coefficients (original + perturbation).
            X_dct = X_nat_dct + eta_dct

            # Inverse transform from DCT coefficients to image.
            X = torch.zeros_like(X_nat)
            for b in range(X_nat.shape[0]):
                for c in range(X_nat.shape[1]):
                    X[b, c] = dct.idct_2d(X_dct[b, c])

            # Input image to the model.
            output = self.model(X, c_trg, mode='enc-dec')

            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()

            # Calculate gradient with respect to DCT coefficients.
            grad_dct = eta_dct.grad

            # Apply gradient only to the selected frequency band.
            grad_dct = grad_dct * freq_mask

            # PGD update: Update DCT coefficients.
            eta_dct_adv = eta_dct.detach() + dct_coef * grad_dct.sign()

            # Limit the perturbation of DCT coefficients.
            eta_dct = torch.clamp(eta_dct_adv, min=-dct_clamp, max=dct_clamp).detach()

            # Apply perturbation only to the selected frequency band.
            eta_dct = eta_dct * freq_mask

        # Restore the final image.
        X_dct_final = X_nat_dct + eta_dct
        X_adv = torch.zeros_like(X_nat)
        for b in range(X_nat.shape[0]):
            for c in range(X_nat.shape[1]):
                X_adv[b, c] = dct.idct_2d(X_dct_final[b, c])

        # Clip image values.
        X_adv = torch.clamp(X_adv, min=-1, max=1)

        return X_adv, X_adv - X_nat


    def create_frequency_masks(self, shape, freq_band='ALL'):
        """
        Function to create masks for different frequency bands.
        
        Args:
            shape: The shape of the mask (batch, channel, height, width).
            freq_band: The frequency band to select ('LOW', 'MID', 'HIGH', 'ALL').
        
        Returns:
            A mask for the selected frequency band (1 for selected area, 0 for unselected area).
        """
        B, C, H, W = shape
        masks = torch.zeros(shape).to(self.device)
        
        # Handle quickly if all frequency bands are selected.
        if freq_band == 'ALL':
            return torch.ones(shape).to(self.device)
        
        # Create a mask for each image and channel.
        for b in range(B):
            for c in range(C):
                # In DCT, (0,0) is the DC component (lowest frequency).
                # The distance from the origin serves as an approximation of the frequency.
                i_coords = torch.arange(H).reshape(-1, 1).repeat(1, W).to(self.device)
                j_coords = torch.arange(W).reshape(1, -1).repeat(H, 1).to(self.device)
                
                # Calculate frequency map (distance from the origin).
                frequency_map = torch.sqrt(i_coords**2 + j_coords**2)
                
                # Normalize the frequency map.
                max_freq = torch.sqrt(torch.tensor((H-1)**2 + (W-1)**2)).to(self.device)
                frequency_map = frequency_map / max_freq
                
                # Divide frequency bands (into 3 parts).
                if freq_band == 'LOW':
                    masks[b, c] = (frequency_map <= 1/3).float()
                elif freq_band == 'MID':
                    masks[b, c] = ((frequency_map > 1/3) & (frequency_map <= 2/3)).float()
                elif freq_band == 'HIGH':
                    masks[b, c] = (frequency_map > 2/3).float()
        
        return masks
