from transformers import TrainerCallback


class SaveTrainingAndEvaluateCallback(TrainerCallback):
    """A custom callback to save training loss at the end of each epoch."""
    
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.epoch_loss = []
        self.metrics = []
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # Try to find the last logged training loss
        last_loss = None
        for entry in reversed(state.log_history):
            if 'loss' in entry:
                    last_loss = entry['loss']
                    break
                
        if last_loss is not None:
            self.epoch_loss.append(last_loss)
            # Directly log to a file
            with open(self.save_path, "a") as f:
                epoch_or_step = f"Epoch {state.epoch}" if state.epoch is not None else f"Step {state.global_step}"
                f.write(f"{epoch_or_step}: Training Loss = {last_loss}\n")
        else:
            # Handle the case where no training loss was found in the log history
            print("Warning: No training loss found for the current epoch.")
        

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # This method is called at the end of each evaluation phase
        print(f"Evaluation: {metrics}")
        if metrics:
            # Write evaluation results to the same file
            with open(self.save_path, "a") as f:
                f.write(f"Epoch {state.epoch}: Evaluation Results = {metrics}\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        # Optionally, summarize the collected losses at the end of training
        print("Training losses per epoch:", self.epoch_loss)
        print("Evaluation: ", self.metrics)