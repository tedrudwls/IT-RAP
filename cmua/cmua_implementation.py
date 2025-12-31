"""
CMUA (Cross-Model Universal Adversarial Watermark) Attack Implementation
Based on: https://github.com/VDIGPKU/CMUA-Watermark

True universal perturbation - one δ optimized across all images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CMUAAttack:
    """
    CMUA Attack for generating universal adversarial perturbations against deepfake generators.
    """
    def __init__(self, epsilon=0.1, step_size=0.01, iterations=20, momentum=0.9, device='cuda'):
        """
        Args:
            epsilon: Maximum perturbation magnitude (L_inf norm)
            step_size: Step size for each iteration
            iterations: Number of attack iterations
            momentum: Momentum factor for gradient updates
            device: Device to run the attack on
        """
        self.epsilon = epsilon
        self.step_size = step_size
        self.iterations = iterations
        self.momentum = momentum
        self.device = device
        self.universal_perturbation = None  # Universal perturbation (shared across images)
    
    def attack_stargan(self, X_nat, generator, c_trg):
        """
        Generate universal adversarial perturbation for StarGAN.
        
        Args:
            X_nat: Clean images (batch) [N, 3, H, W]
            generator: StarGAN generator
            c_trg: Target attribute labels [N, c_dim]
            
        Returns:
            X_adv: Adversarial images
            perturbation: Universal perturbation
        """
        batch_size = X_nat.size(0)
        
        # ✅ Initialize ONE universal perturbation (not per-image)
        if self.universal_perturbation is None:
            self.universal_perturbation = torch.zeros(1, 3, X_nat.size(2), X_nat.size(3)).to(self.device)
        
        delta = self.universal_perturbation.clone()
        delta.requires_grad = True
        
        # Momentum buffer for gradient updates
        momentum_grad = torch.zeros_like(delta)
        
        # ✅ Generate original outputs once (for all images)
        with torch.no_grad():
            original_gen_list = []
            for i in range(batch_size):
                original_gen, _ = generator(X_nat[i:i+1], c_trg[i:i+1])
                original_gen_list.append(original_gen)
            original_gen_batch = torch.cat(original_gen_list, dim=0)
        
        print(f"[CMUA] Starting universal perturbation optimization...")
        print(f"[CMUA] Training on {batch_size} image-attribute pairs")
        
        for iteration in range(self.iterations):
            # ✅ Apply the SAME delta to ALL images
            X_perturbed = X_nat + delta.expand(batch_size, -1, -1, -1)
            X_perturbed = torch.clamp(X_perturbed, -1, 1)
            
            # ✅ Generate perturbed outputs for all images
            perturbed_gen_list = []
            for i in range(batch_size):
                perturbed_gen, _ = generator(X_perturbed[i:i+1], c_trg[i:i+1])
                perturbed_gen_list.append(perturbed_gen)
            perturbed_gen_batch = torch.cat(perturbed_gen_list, dim=0)
            
            # ✅ CMUA Loss: MAXIMIZE MSE between generated images (aggregate across all images)
            loss = torch.nn.MSELoss()(perturbed_gen_batch, original_gen_batch.detach())
            
            # ✅ Backward to get gradients w.r.t. delta (NOT w.r.t. individual images)
            generator.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            grad = delta.grad
            
            # ✅ Momentum-based gradient update
            momentum_grad = self.momentum * momentum_grad + grad / (grad.abs().mean() + 1e-8)
            
            # ✅ Update delta (gradient ascent to maximize loss)
            with torch.no_grad():
                delta = delta + self.step_size * momentum_grad.sign()
                
                # Project to epsilon ball
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            
            delta = delta.detach()
            delta.requires_grad = True
            
            if (iteration + 1) % 5 == 0:
                print(f"[CMUA] Iteration {iteration+1}/{self.iterations}, Loss: {loss.item():.4f}")
        
        # ✅ Save the final universal perturbation
        self.universal_perturbation = delta.detach()
        
        # Return adversarial images for the current batch
        X_adv = X_nat + self.universal_perturbation.expand(batch_size, -1, -1, -1)
        X_adv = torch.clamp(X_adv, -1, 1)
        
        return X_adv, self.universal_perturbation
    
    def get_universal_perturbation(self):
        """
        Get the current universal perturbation.
        
        Returns:
            universal_perturbation: The universal perturbation tensor
        """
        return self.universal_perturbation
    
    def apply_universal_perturbation(self, X_nat):
        """
        Apply the universal perturbation to clean images.
        
        Args:
            X_nat: Clean images
            
        Returns:
            X_adv: Adversarial images with universal perturbation applied
        """
        if self.universal_perturbation is None:
            raise ValueError("Universal perturbation not generated yet. Run attack_stargan first.")
        
        # Expand universal perturbation to match batch size
        batch_size = X_nat.size(0)
        perturbation = self.universal_perturbation.expand(batch_size, -1, -1, -1)
        
        # Apply perturbation
        X_adv = X_nat + perturbation
        
        # Clamp to valid image range
        X_adv = torch.clamp(X_adv, -1, 1)
        
        return X_adv
    
    def reset(self):
        """
        Reset the universal perturbation.
        """
        self.universal_perturbation = None


class CMUAAttackMultiModel:
    """
    CMUA Attack for multiple deepfake models (cross-model universal attack).
    """
    def __init__(self, epsilon=0.1, step_size=0.01, iterations=20, momentum=0.9, device='cuda'):
        """
        Args:
            epsilon: Maximum perturbation magnitude (L_inf norm)
            step_size: Step size for each iteration
            iterations: Number of attack iterations
            momentum: Momentum factor for gradient updates
            device: Device to run the attack on
        """
        self.epsilon = epsilon
        self.step_size = step_size
        self.iterations = iterations
        self.momentum = momentum
        self.device = device
        self.universal_perturbation = None
    
    def attack_multi_models(self, X_nat, generators, c_trg_list):
        """
        Generate universal adversarial perturbation for multiple generators.
        
        Args:
            X_nat: Clean images (batch)
            generators: List of generators to attack
            c_trg_list: List of target attribute labels for each generator
            
        Returns:
            X_adv: Adversarial images
            perturbation: Universal perturbation
        """
        batch_size = X_nat.size(0)
        
        # Initialize ONE universal perturbation
        if self.universal_perturbation is None:
            self.universal_perturbation = torch.zeros(1, 3, X_nat.size(2), X_nat.size(3)).to(self.device)
        
        delta = self.universal_perturbation.clone()
        delta.requires_grad = True
        
        momentum_grad = torch.zeros_like(delta)
        
        # Generate original outputs for all models
        original_gens_all_models = []
        with torch.no_grad():
            for generator, c_trg in zip(generators, c_trg_list):
                original_gen_list = []
                for i in range(batch_size):
                    original_gen, _ = generator(X_nat[i:i+1], c_trg[i:i+1])
                    original_gen_list.append(original_gen)
                original_gens_all_models.append(torch.cat(original_gen_list, dim=0))
        
        print(f"[CMUA Multi-Model] Training on {len(generators)} models with {batch_size} image pairs")
        
        for iteration in range(self.iterations):
            # Apply the SAME delta to ALL images
            X_perturbed = X_nat + delta.expand(batch_size, -1, -1, -1)
            X_perturbed = torch.clamp(X_perturbed, -1, 1)
            
            # Accumulate loss from all models
            total_loss = 0
            for idx, (generator, c_trg, original_gen_batch) in enumerate(zip(generators, c_trg_list, original_gens_all_models)):
                # Generate perturbed outputs
                perturbed_gen_list = []
                for i in range(batch_size):
                    perturbed_gen, _ = generator(X_perturbed[i:i+1], c_trg[i:i+1])
                    perturbed_gen_list.append(perturbed_gen)
                perturbed_gen_batch = torch.cat(perturbed_gen_list, dim=0)
                
                # CMUA Loss
                loss = torch.nn.MSELoss()(perturbed_gen_batch, original_gen_batch.detach())
                total_loss += loss
            
            # Average loss across models
            total_loss = total_loss / len(generators)
            
            # Backward
            for generator in generators:
                generator.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            total_loss.backward()
            grad = delta.grad
            
            # Momentum update
            momentum_grad = self.momentum * momentum_grad + grad / (grad.abs().mean() + 1e-8)
            
            # Update delta
            with torch.no_grad():
                delta = delta + self.step_size * momentum_grad.sign()
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            
            delta = delta.detach()
            delta.requires_grad = True
            
            if (iteration + 1) % 5 == 0:
                print(f"[CMUA Multi-Model] Iteration {iteration+1}/{self.iterations}, Avg Loss: {total_loss.item():.4f}")
        
        # Save final perturbation
        self.universal_perturbation = delta.detach()
        
        X_adv = X_nat + self.universal_perturbation.expand(batch_size, -1, -1, -1)
        X_adv = torch.clamp(X_adv, -1, 1)
        
        return X_adv, self.universal_perturbation
    
    def get_universal_perturbation(self):
        """Get the current universal perturbation."""
        return self.universal_perturbation
    
    def apply_universal_perturbation(self, X_nat):
        """Apply the universal perturbation to clean images."""
        if self.universal_perturbation is None:
            raise ValueError("Universal perturbation not generated yet.")
        
        batch_size = X_nat.size(0)
        perturbation = self.universal_perturbation.expand(batch_size, -1, -1, -1)
        X_adv = X_nat + perturbation
        X_adv = torch.clamp(X_adv, -1, 1)
        
        return X_adv
    
    def reset(self):
        """Reset the universal perturbation."""
        self.universal_perturbation = None
