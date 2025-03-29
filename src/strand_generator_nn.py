import os
import torch
import torch.nn as nn
from strand_evaluator_nn import EvaluatorModel
import torch.optim as optim

class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # We define a network that expects a flattened vector of size 40.
        self.stack = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # Output a delta (dx, dy, dz)
        )
    
    def forward(self, x):
        """
        x: Tensor of shape (5,8) where each row is 
           (dx, dy, dz, a, c, g, u, -).
           
        The generator iteratively computes a delta (dx, dy, dz) from the entire cluster,
        then adds that delta to the first three features of all non-base nucleotides.
        The base nucleotide (first row) remains fixed.
        """
        max_iter = 5
        # Save the base nucleotide (first row) so it remains unchanged
        base = x[0:1, :].clone()  # shape: (1, 8)
        # Start with the initial cluster
        cluster = x.clone()
        for _ in range(max_iter):
            # Flatten the entire cluster into a 40-element vector
            flat = cluster.view(-1)  # shape: (40,)
            delta = self.stack(flat)  # shape: (3,)
            # Update all nucleotides except the base:
            updated_rest = cluster[1:, :].clone()  # shape: (4, 8)
            # Add delta to the first three features (dx, dy, dz) of each nucleotide in updated_rest.
            updated_rest[:, :3] = updated_rest[:, :3] + delta.unsqueeze(0)
            # Reassemble the cluster with the fixed base.
            cluster = torch.cat([base, updated_rest], dim=0)
        return cluster


# ----------------- Adversarial Training Setup -----------------
def adversarial_training(generator, evaluator, num_steps=100, lr=1e-3):
    """
    Trains the generator to fool the evaluator. The evaluator is already trained and loaded.
    The generator's objective is to have the evaluator classify its output as 'real'.
    If the evaluator was trained with is_fake == 0 indicating real, then the target is 0.
    """
    # Freeze evaluator parameters
    evaluator.eval()
    for param in evaluator.parameters():
        param.requires_grad = False

    # Set generator to train mode and create an optimizer for it
    generator.train()
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Evaluator outputs a logit

    # For demonstration, we create a single sample cluster (5,8).
    # In practice, you would use a batch of clusters.
    sample_cluster = torch.randn(5, 8)
    sample_cluster = sample_cluster  # shape: (5,8)

    # Target label: if your evaluator was trained with is_fake==0 for real, then:
    target = torch.zeros(1, 1)  # shape (1,1) to match evaluator output

    for step in range(num_steps):
        optimizer.zero_grad()
        # Generator refines the cluster
        refined_cluster = generator(sample_cluster)
        # If refined_cluster is (5,8) and evaluator expects batch dim, unsqueeze:
        if refined_cluster.dim() == 2:
            refined_cluster = refined_cluster.unsqueeze(0)  # now (1,5,8)
        # Evaluator scores the refined cluster
        eval_output = evaluator(refined_cluster)  # shape: (1,1)
        # Generator loss: want evaluator to output a logit near 0 (indicating real)
        loss = criterion(eval_output, target)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, Evaluator Output = {eval_output.item():.4f}")

    # Return the final refined cluster for inspection
    return generator(sample_cluster)

# ----------------- Main -----------------
if __name__ == "__main__":
    # Set dir to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Load the pretrained evaluator model from the provided path
    evaluator_path = "models/evaluator_model.pth"
    evaluator = EvaluatorModel()
    evaluator.load_state_dict(torch.load(evaluator_path))
    evaluator.eval()

    # Initialize the generator model
    generator = GeneratorModel()

    # Train the generator to try to trick the evaluator
    print("Starting adversarial training of generator...")
    final_cluster = adversarial_training(generator, evaluator, num_steps=100, lr=1e-3)

    # Evaluate final output
    # Make sure to add a batch dimension for the evaluator if needed:
    final_input = final_cluster if final_cluster.dim() == 3 else final_cluster.unsqueeze(0)
    final_eval_output = evaluator(final_input)
    print("\nFinal Refined Cluster:\n", final_cluster)
    print("Final Evaluator Output (logit):", final_eval_output.item())
