import os
import torch
import torch.nn as nn
from strand_evaluator_nn import EvaluatorModel
from strand_generator_nn import GeneratorModel
from tools import *

# ----------------- Train Using Evaluator Model -----------------
def train_generator_with_evaluator(generator, evaluator, num_steps=100, lr=1e-3):
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

# TODO: Add some sort of freezing mechanism when one side is winning too much
# eg. Generator loss is too low, freeze it for a while and let the evaluator catch up


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
    final_cluster = train_generator_with_evaluator(generator, evaluator, num_steps=10000, lr=1e-5)

    # Evaluate final output
    # Make sure to add a batch dimension for the evaluator if needed:
    final_input = final_cluster if final_cluster.dim() == 3 else final_cluster.unsqueeze(0)
    final_eval_output = evaluator(final_input)
    print("\nFinal Refined Cluster:\n", final_cluster)
    print("Final Evaluator Output (logit):", final_eval_output.item())
    
    # Save the generator model
    torch.save(generator.state_dict(), "models/generator_model.pth")
    
    print("Generator model saved as 'models/generator_model.pth'")
    
    # load the generator model
    generator = GeneratorModel()
    generator.sequence_to_pdb("ACGUACGUACGUA")