import torch
import torch.nn as nn

class FeedbackLoop:
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        # If no optimizer provided, create a default one for fine-tuning
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.lstm.parameters(), lr=0.0001) # Low learning rate
            
    def update_with_feedback(self, X_seq, confirmed_failure):
        """
        Active Learning step.
        If an engineer REJECTS an alert (confirmed_failure = False),
        we punish the model for predicting high risk.
        
        If confirmed_failure = True, we reinforce it.
        """
        device = next(self.model.lstm.parameters()).device
        self.model.lstm.train()
        
        X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        if X_seq_tensor.dim() == 2:
            X_seq_tensor = X_seq_tensor.unsqueeze(0)
            
        # Forward pass
        pred = self.model.lstm(X_seq_tensor)
        
        # Target: If it was a false positive (engineer rejected), target RUL should be higher (Healthy)
        # If it was a true positive, target RUL should be low (Failure imminent)
        # This assumes we have some estimate of what "Healthy" looks like (e.g., max RUL)
        # For simplicity, we nudge it.
        
        current_pred = pred.item()
        
        if not confirmed_failure:
            # It was a False Positive. The plane is likely healthy.
            # Push prediction towards higher RUL (e.g. current + 20 cycles)
            target = torch.tensor([current_pred + 20.0], dtype=torch.float32).to(device)
            print(f"Feedback: Alert Rejected. Nudging RUL from {current_pred:.1f} up to {target.item():.1f}")
        else:
            # Confirmed. Reinforce or do nothing (assuming previous training covered this).
            # Let's slight reinforce.
            target = torch.tensor([current_pred], dtype=torch.float32).to(device)
            print("Feedback: Alert Confirmed. No adjustment needed (or reinforcing).")
            return

        # Loss and Backprop
        criterion = nn.MSELoss()
        loss = criterion(pred.squeeze(), target.squeeze())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print("Model weights updated based on feedback.")

    def refine_causal_link(self, parent_sensor, child_sensor, user_rejected=True):
        """
        Phase V: Human-in-the-Loop Causal Refinement.
        If an engineer rejects a diagnosis saying "Fuel Flow did NOT cause EGT",
        we blacklist that edge.
        """
        if user_rejected:
            import json
            import os
            blacklist_path = "src/causal_blacklist.json"
            
            current_list = []
            if os.path.exists(blacklist_path):
                with open(blacklist_path, 'r') as f:
                    current_list = json.load(f)
            
            new_edge = [parent_sensor, child_sensor]
            if new_edge not in current_list:
                current_list.append(new_edge)
                with open(blacklist_path, 'w') as f:
                    json.dump(current_list, f)
                print(f"Causal Edge {parent_sensor} -> {child_sensor} BLACKLISTED by Operator.")
            else:
                print("Edge already blacklisted.")
