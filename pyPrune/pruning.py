import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from typing import Optional, Callable
from pyPrune.utils import *
import numpy as np
import pickle
from tqdm import tqdm
import logging
import json
from pyPrune.utils import *
import datetime

# if log file already exists load it and append else create a new one

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class IterativeMagnitudePruning:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,  
                 steps: list, optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                 pruning_criterion: Optional[Callable[[float, torch.Tensor], torch.Tensor]] = None,
                 device: Optional[str] = None, save_dir: str = 'pruning_checkpoints', finetune_epochs: int = 0, 
                 pretrain_epochs: int = 0, learning_rate: float = 0.01, file_handler: str = "logger.log",
                 prunable_layers: tuple = (torch.nn.Conv2d,torch.nn.Linear)) -> None:
        """
        Initializes the IterativeMagnitudePruning class to perform iterative magnitude pruning on a neural network model.

        This constructor sets up the model, dataset loaders, pruning parameters, optimizer, and device configuration 
        necessary for the pruning process. It also ensures that the save directory exists for storing model checkpoints 
        during pruning, and initializes the model's weights and metrics tracking.

        Parameters:
            model (nn.Module): The neural network model to be pruned.
            train_loader (DataLoader): The DataLoader instance used for training the model.
            test_loader (DataLoader): The DataLoader instance used for evaluating the model after pruning.
            steps (list): A list of sparsity levels to prune the model to.
            optimizer (torch.optim.Optimizer): The optimizer to be used for fine-tuning the pruned model.
            criterion (nn.Module): The loss function (e.g., CrossEntropyLoss) to be used during training and evaluation.
            pruning_criterion (Optional[Callable[[float, torch.Tensor], torch.Tensor]], optional): A custom function for pruning 
                based on specific criteria. Defaults to None, using magnitude-based pruning.
            device (Optional[str], optional): The device to run the model on ('cpu' or 'cuda'). Defaults to None, which chooses 'cuda' 
                if available.
            save_dir (str, optional): Directory where pruned model checkpoints will be saved. Defaults to 'pruning_checkpoints'.
            finetune_epochs (int, optional): Number of epochs for fine-tuning the model after each pruning step. Defaults to 0.
            pretrain_epochs (int, optional): Number of epochs for pretraining the model before starting pruning. Defaults to 0.
            learning_rate (float, optional): The learning rate used during fine-tuning. Defaults to 0.01.

        Returns:
            None: The constructor initializes the class and does not return any value.

        Example:
            model = MyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            train_loader = DataLoader(...)
            test_loader = DataLoader(...)

            pruner = IterativeMagnitudePruning(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                final_sparsity=0.9,
                steps=10,
                optimizer=optimizer,
                criterion=criterion,
                pretrain_epochs=5,
                finetune_epochs=2,
                learning_rate=0.01
            )
        """
        self.prunable_layers = prunable_layers
        self.save_dir = save_dir
    
        self.pickle_name = f"{self.save_dir}/pruner.pkl"
        self.setup_save_dir()
        
        self.steps =  steps
        self.pruning_criterion = pruning_criterion or self.magnitude_prune
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.finetune_epochs = finetune_epochs
        self.pretrain_epochs = pretrain_epochs
        
        self.current_finetune_epoch = 0
        self.current_pretrain_epoch = 0
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.learning_rate = learning_rate
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.current_sparsity = 0.0
        self.best_model_weights = None


        self.initial_parameters = self.save_initial_parameters()

        self.weight_history = [self.initial_parameters]
        self.metrics = {
            'sparsity': [],
            'loss': [],
            'accuracy': [],
            'gradients': [],
            'optimizer': [],
            'step': [],
        }

        # if steps has the first index as 0, update pretrain_epochs accordingly
        if self.steps[0] == 0 and self.pretrain_epochs == 0:
            self.pretrain_epochs = self.pretrain_epochs + 1  # Set pretrain_epochs to 1 more
            self.steps = self.steps[1:]  # Remove the first element (0) from steps
        
        # Log the initialization details with a file handler
        if os.path.exists(self.save_dir + "/" + file_handler):
            # get the current time and append it to the file name
            file_handler = file_handler.split(".")
            file_handler = file_handler[0] + "_" + str(datetime.datetime.now()) + "." + file_handler[1]
        file_handler = logging.FileHandler(self.save_dir + "/" + file_handler)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        self.logger = logger
        self.complete = False
        
        logger.info("IterativeMagnitudePruning initialized.")
        logger.info(f"Device: {self.device}, Final sparsity: {self.steps[-1]}, Steps: {self.steps}")

    def delete_pickle(self) -> None:
        """
        Deletes all pickle files created during the pruning process.
        
        This method deletes all pickle files created during the pruning process. It is useful
        """
        if os.path.exists(self.pickle_name):
            os.remove(self.pickle_name)
            logger.info(f"Deleted pickle file: {self.pickle_name}")
        else:
            logger.info(f"No pickle file found at: {self.pickle_name}")
            
    def setup_save_dir(self) -> None:
        """
        Sets up the directory where model checkpoints will be saved during the pruning process.

        This method checks if the specified save directory exists. If not, it creates the directory to ensure that 
        model checkpoints can be saved during the pruning process. A log message is generated to inform the user 
        about the creation or existence of the directory.

        Parameters:
            None: This method does not accept any parameters as it operates on the instance's `self.save_dir`.

        Returns:
            None: This method does not return any value. It only ensures the save directory exists.

        Example:
            pruner = IterativeMagnitudePruning(...)
            pruner.setup_save_dir()
            # This will create the directory specified by `self.save_dir` if it doesn't already exist.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"Created directory: {self.save_dir}")
        else:
            logger.info(f"Save directory {self.save_dir} already exists.")

        # initalize the pickle
        with open(self.pickle_name, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Init Pruner saved as pickle.")

    def save_initial_parameters(self) -> dict:
        """
        Saves the initial values of the model's weight parameters any pruning 
        is applied. This method iterates over the model's parameters, 
        and stores their initial values in a dictionary. The initial parameters are saved 
        for potential restoration after pruning or to track changes during the pruning process.

        Returns:
            dict: 
                A dictionary containing the initial values of the model's parameters. 
                The dictionary maps parameter names (as strings) to their corresponding 
                values (as torch.Tensor objects).

        Example:
            # Create an instance of IterativeMagnitudePruning
            pruning = IterativeMagnitudePruning(...)

            # Save initial weights
            initial_weights = pruning.save_initial_parameters()

            # The initial_weights dictionary can be used for comparison or restoring weights.
        """       

        initial_parameters = {}
        for name, param in self.model.named_parameters(): #keep batchnorm weights here for rewinding
            initial_parameters[name] = param.data.clone()
        logger.info(f"Initial values saved for {len(initial_parameters)}  parameters.")
        return initial_parameters

    def unroll(self, percentage: float = 0) -> None:
        """
        Unrolls the model's weight parameters to facilitate pruning based on their magnitudes.

        This method extracts all weight parameters from the model, and flattens them into a single tensor. It calculates 
        the number of weights to prune based on the specified percentage and returns the relevant data 
        for pruning.

        Parameters:
            percentage (float): The percentage of weights to prune from the model. Should be between 0 and 1.
                                This value determines the sparsity level to which the model will be pruned.
                                For example, 0.1 means 10% of weights will be pruned.

        Returns:
            tuple: A tuple containing:
                - num_prune (int): The number of weights to prune based on the specified percentage.
                - all_weights (torch.Tensor): A flattened tensor containing all model weights.

        Example:
            pruner = IterativeMagnitudePruning(...)
            num_prune, all_weights, = pruner.unroll(percentage=0.1)
        """

        logger.debug(f"Unrolling model at {percentage * 100:.2f}% sparsity")

        all_weights = []
        for module in get_pruneable_modules(self.model, self.prunable_layers):
                all_weights.append(module.weight.data.flatten())

        all_weights = torch.cat(all_weights)

        num_prune = max(1, int(all_weights.numel() * percentage))
        
        return num_prune, all_weights

    def update_optimizer(self) -> None:
        """
        Resets the optimizer each time training restarts (after each pass of pruning). This will clear 
        any adaptive learning rates, momentums, rmsprop, etc... from the optimizer

        Parameters:
            None

        Returns:
            None

        Example:
            pruner = IterativeMagnitudePruning(...)
            pruner.update_optimizer()
            # This updates the optimizer after pruning to ensure that it works only with trainable parameters.
        """
        #re-initialize the optimizer
        self.optimizer = self.optimizer.__class__(self.model.parameters(), lr=self.learning_rate)

        # Log the number of parameters being passed to the optimizer
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Optimizer updated to reflect {len(list(self.model.parameters()))} parameters, Total: {total_params} parameters.")

        # Log the parameters (optional, can be verbose for large models)
        logger.debug(f"Optimizer parameters: {[p.shape for p in self.model.parameters()]}")

    def save_checkpoint(self, step: int, file_path: str) -> None:
        """
        Saves the model and optimizer state to a checkpoint file.

        This method serializes the current state of the model and optimizer to a checkpoint 
        file, allowing for later recovery of the model's state at a particular pruning step. 
        The checkpoint contains the model's state dictionary, optimizer's state dictionary, 
        and the current step in the pruning process. It helps in resuming training or pruning 
        from a specific point in case of interruption or for analysis.

        Parameters:
            step (int): The current pruning step number, used to name the checkpoint file.
            file_path (str): The path to the file where the checkpoint will be saved.

        Returns:
            None

        Example:
            pruner = IterativeMagnitudePruning(...)
            pruner.save_checkpoint(step=3, file_path="pruning_checkpoints/step_3.pth")
            # Saves the model and optimizer states at step 3 to 'step_3.pth'.
        """

        try:
            checkpoint = {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, file_path)
            logger.info(f"Checkpoint saved at {file_path}")

        except KeyError as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def magnitude_prune(self, percentage: float) -> None:
        """
        Prunes the model's weights based on their magnitudes.

        This method performs magnitude-based pruning, which removes weights from the model 
        that have the smallest absolute values. The pruning is applied to the model’s 
        parameters by zeroing out the weights with the smallest magnitudes, effectively 
        reducing the number of active parameters. The method also updates the optimizer to 
        reflect the pruned weights.

        Parameters:
            percentage (float): The percentage of weights to prune, represented as a decimal 
                                between 0 and 1. For example, 0.2 corresponds to pruning 20% 
                                of the smallest magnitude weights.

        Returns:
            None

        Example:
            pruner = IterativeMagnitudePruning(...)
            pruner.magnitude_prune(percentage=0.2)
            # Prunes 20% of the smallest magnitude weights from the model.
        """

        logger.info(f"Pruning model at {percentage * 100:.2f}% sparsity.")
        num_prune, all_weights = self.unroll(percentage)
        # O(n) partition to select the threshold. the element at position num_prune-1 in the partitioned array is 
        # the pruning threshold. partition is on absolute value of weights
        # needs to be copied back to cpu for numpy to work, maybe not worth it compared to sorting
        threshold_value = np.partition(np.abs(all_weights.cpu().numpy()),num_prune-1)[num_prune-1]
        if num_prune == 0:
            threshold_value = float('inf')
        # Apply pruning: directly zero out weights below the threshold
        for module in get_pruneable_modules(self.model, self.prunable_layers):
                mask = torch.abs(module.weight.data) >= threshold_value
                module.weight.data.mul_(mask.float())  # Prune weights

                # Detach pruned weights from gradients
                module.weight.grad = None  # Zero out any existing gradient for the pruned weights
                module.weight.requires_grad = not module.weight.data.eq(0).all()  # Set requires_grad to False if all weights are zero

        logger.debug(f"Pruning applied at {percentage * 100:.2f}% sparsity with threshold {threshold_value:.6f}")
        self.update_optimizer()

    def reset_weights(self) -> None:
        """
        Resets the model's weights to their initial values.

        This method restores the weights of the model to the state they were in when the 
        `IterativeMagnitudePruning` instance was first initialized. It is useful for cases 
        where you want to undo the pruning process and start with the original weights.

        Parameters:
            None

        Returns:
            None

        Example:
            pruner = IterativeMagnitudePruning(...)
            pruner.reset_weights()
            # The model's weights are now restored to their initial values.
        """

        for name, param in self.model.named_parameters(): #keep batch norm
            param.data = self.initial_parameters[name].clone()
        logger.info("Weights reset to initial values.")

    def update_metrics(self, loss: float, accuracy: float, gradients: torch.Tensor) -> None:
        """
        Updates the pruning metrics for the current step.

        This method updates the internal `metrics` dictionary by appending the current 
        loss, accuracy, and gradients (if provided) for the pruning process. The metrics 
        are used to track the performance of the model during the pruning and fine-tuning 
        steps.

        Parameters:
            loss (float): The loss value for the current training or evaluation step.
            accuracy (float): The accuracy of the model during the current step.
            gradients (torch.Tensor, optional): The gradients of the model parameters. 
                This can be `None` during evaluation steps.

        Returns:
            None

        Example:
            pruner.update_metrics(loss.item(), accuracy, next(pruner.model.parameters()).grad)
            # The metrics dictionary is now updated with the new loss, accuracy, and gradients.
        """

        # Record metrics
        self.metrics['sparsity'].append(self.current_sparsity)
        self.metrics['loss'].append(loss)
        self.metrics['accuracy'].append(accuracy)

        # check accuracy if sparisty is greater than zero and best accuracy update the best model
        if self.current_sparsity > 0 and accuracy > max(self.metrics['accuracy']):
            logger.info(f"Updating best model weights at {self.current_sparsity * 100:.2f}% sparsity with accuracy {accuracy:.2f} %")
            self.best_model_weights = self.model.state_dict()

    def epoch(self, type: str = "train", patience: int = 5) -> None:
        """
        Trains or evaluates the model depending on the specified mode, with early stopping.

        This method controls whether the model is in training or evaluation mode. 
        It performs training by iterating over the training data and updating the model 
        parameters using backpropagation and the optimizer. During evaluation, it 
        computes the average loss and accuracy on the test data without updating the model.
        Early stopping is applied based on validation loss.

        Parameters:
            type (str): The mode of the operation. It can either be "train" or "eval".
                - "train" will train the model on the training set.
                - "eval" will evaluate the model on the test set.
            patience (int): The number of epochs to wait for improvement in validation loss 
                            before stopping the training early.

        Returns:
            None

        Example:
            pruner.train("train")  # Runs the training loop on the model using the training data.
            pruner.train("eval")   # Runs the evaluation loop on the model using the test data.

        Note:
            In training mode, the gradients of zeroed-out weights are masked during the 
            backpropagation step to ensure that pruned weights are not updated.
        """

        if type == "train":
            self.model.train()
            epochs_without_improvement = 0
            best_val_loss = float("inf")

            for data, target in tqdm(self.train_loader, desc="Training", unit="batch"):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # Mask the gradients for zeroed-out weights
                for module in get_pruneable_modules(self.model, self.prunable_layers):
                    if module.weight.grad is not None:
                        mask = module.weight.data != 0  # Mask for non-zero weights
                        module.weight.grad *= mask.float()  # Zero out gradients for pruned weights

                self.optimizer.step()
                accuracy = 100. * output.argmax(dim=1).eq(target).sum().item() / target.size(0)
                
            # Perform evaluation to get validation loss
            self.model.eval()
            total_loss = 0.0
            correct = 0
            with torch.no_grad():
                for data, target in tqdm(self.test_loader, desc="Evaluating", unit="batch"):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    total_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            total_loss /= len(self.test_loader.dataset)
            accuracy = 100. * correct / len(self.test_loader.dataset)

            # Early stopping logic
            if total_loss < best_val_loss:
                best_val_loss = total_loss
                epochs_without_improvement = 0  # Reset the counter if there's improvement
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
                return  # Stop training if no improvement for 'patience' epochs

            # Update the metrics dictionary
            self.update_metrics(total_loss, accuracy, None)
            logger.info(f"Training step complete, Loss: {loss.item()}")

        elif type == "eval":
            self.model.eval()
            total_loss = 0.0
            correct = 0
            with torch.no_grad():
                for data, target in tqdm(self.test_loader, desc="Evaluating", unit="batch"):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    total_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            total_loss /= len(self.test_loader.dataset)
            accuracy = 100. * correct / len(self.test_loader.dataset)

            # Update the metrics dictionary
            self.update_metrics(total_loss, accuracy, None)
            logger.info(f"Evaluation complete, Average loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

        logger.debug(f"Metrics: {self.metrics}")
        logger.debug(f"Current sparsity: {self.current_sparsity}")
        logger.debug(f"Accuracy: {accuracy}")

        clean_memory()  # Clean up memory after each epoch

    def update_pickle(self) -> None:
        """
        Updates the pickle file with the current state of the pruner object.

        This method saves the current state of the pruner object to a pickle file. 
        It is useful for saving the pruner object during the pruning process, allowing 
        the process to be resumed or analyzed later.

        Parameters:
            None

        Returns:
            None

        Example:
            pruner.update_pickle()
            # The current state of the pruner object is saved to the pickle file.
        """

        with open(self.pickle_name, 'wb') as f:
            pickle.dump(self, f)
        logger.info("Pruner saved as pickle.")

    def run(self) -> None:
        """
        Runs the iterative magnitude pruning process, including pretraining, pruning, 
        and fine-tuning the model.

        The method begins by saving the initial model weights and optionally performs 
        pretraining (if specified). It then proceeds to iteratively prune the model 
        over a specified number of steps. After each pruning step, the model is 
        fine-tuned to recover any performance lost due to pruning. The process is 
        logged and checkpoints are saved at each pruning step. Finally, the pruning 
        metrics are saved to a JSON file.

        Steps:
        1. Save the initial weights of the model.
        2. Optionally pretrain the model for the specified number of epochs.
        3. Update the initial weights to be the rewound weights.
        4. Perform iterative pruning over a defined number of steps, gradually 
        increasing sparsity.
        5. Fine-tune the model after each pruning step to recover performance.
        6. Save a checkpoint after each pruning step.
        7. Evaluate the model performance after each pruning and fine-tuning step.
        8. Reset the model weights to the rewinded state after each pruning step.
        7. Save all pruning metrics to a JSON file.

        Parameters:
            None

        Returns:
            None

        Example:
            pruner.run()  # Runs the entire pruning process, including pretraining,
                        # pruning, fine-tuning, and saving metrics.

        Note:
            - The final sparsity level is specified when initializing the `IterativeMagnitudePruning` object.
            - The method will handle the saving of model checkpoints and metrics.
            - If the number of pretrain epochs is greater than 0, pretraining will be done before pruning starts.
        """
        try:
            self.initial_parameters = self.save_initial_parameters()

            if self.pretrain_epochs > 0:
                logger.info("Starting pretraining...")
                for pretrain_epoch_steps in range(self.pretrain_epochs):
                    if pretrain_epoch_steps <= self.pretrain_epochs:
                        logger.info(f"model already pre-trained at step {pretrain_epoch_steps}, skipping...")
                    else:
                        logger.info(f"Pretraining the model at step {pretrain_epoch_steps + 1}...")
                        self.pretrain_epochs = pretrain_epoch_steps
                        logger.info(f"Pretraining the model at step {pretrain_epoch_steps + 1}...")
                        self.epoch("train")
                    
            # Update initial weights after pretraining for the "rewinding" effect
            self.initial_parameters = self.save_initial_parameters()
            self.best_model_weights = self.model.state_dict()
            logger.info(f"Starting pruning with {self.steps} steps...")

            for step in tqdm(self.steps, desc="Pruning Steps", unit="step"):
                if step <= self.current_finetune_epoch:
                    logger.info(f"model already pruned at step {step}, skipping...")
                else:
                    self.current_finetune_epoch = step
                    logger.info(f"Starting pruning step: {step * 100:.2f}% sparsity")

                    # Retrain the model  
                    logger.info("Fine-tuning the model...")
                    if self.finetune_epochs > 0:    
                        for finetune_epoch_steps in range(self.finetune_epochs):
                            logger.info(f"Fine-tuning the model at step {finetune_epoch_steps + 1}...")
                            self.epoch("train")

                    # Prune the model
                    logger.info(f"Pruning the model at {step * 100:.2f}% sparsity...")
                    self.magnitude_prune(step)

                    # Update the current sparsity
                    self.assert_sparsity(step) 

                    logger.info("Updating optimizer to reflect pruned weights...")
                    self.save_checkpoint(step, os.path.join(self.save_dir, f"pruned_model_step_{float(step):.4f}.pth"))
                    
                    # Add metrics for pruning step
                    self.metrics['step'].append(step)
                    
                    
                    self.epoch("eval")
                    # save the current weights
                    logger.info(f"Saving weights at {step * 100:.2f}% sparsity...")
                    self.weight_history.append(self.model.state_dict())

                    # reset to the rewind weights
                    logger.info(f"Resetting weights to rewind state at {self.pretrain_epochs}, currently at  {step * 100:.2f}% sparsity...")
                    self.reset_weights()
                    logger.info(f"Pickling at {step * 100:.2f}% sparsity...")
                    self.update_pickle()
                    print("\n\n\n")

            logger.info("Pruning complete.")
            # save the metrics as a json
            logger.info("Saving metrics...")
            self.save_metrics()
            logger.info("Metrics saved to pruning_metrics.json")     

            self.update_pickle()
            logger.info("Pruner saved as pickle.")

            logger.info("Pruning process complete.")
            self.complete = True
        except Exception as e:
            logger.error(f"Error during pruning: {str(e)}")
            logger.error("Deleting pickle file...")
            self.delete_pickle()

    def save_metrics(self) -> None:
        """
        Saves the pruning process metrics to a JSON file. The metrics include sparsity, 
        loss, accuracy, gradients, optimizer state, and step information collected 
        throughout the iterative pruning process.

        The method serializes the metrics dictionary, converting any tensors or 
        parameters into native Python types (e.g., scalars or lists). After conversion, 
        it saves the metrics into a JSON file at the specified `save_dir`.

        Steps:
        1. Convert all tensors in the metrics dictionary to native Python types 
        (scalars or lists).
        2. Serialize the metrics into a JSON-compatible format.
        3. Write the serialized metrics to a JSON file in the `save_dir`.

        Parameters:
            None

        Returns:
            None

        Example:
            pruner.save_metrics()  # Saves the current pruning metrics to the 
                                # 'pruning_metrics.json' file in the save directory.

        Note:
            - This method is typically called after the pruning process is complete.
            - The metrics will be saved in a JSON format, where all tensor data 
            is converted to standard Python types.
        """
        # Apply the conversion to the entire metrics dictionary
        metrics_serializable = {key: [self.convert_tensor(val) for val in values] for key, values in self.metrics.items()}

        # Save the metrics as a JSON file
        with open(os.path.join(self.save_dir, 'pruning_metrics.json'), 'w') as f:
            json.dump(metrics_serializable, f)
        logger.info("Metrics saved to pruning_metrics.json")
    
    def convert_tensor(self, t):
        """
        Converts a PyTorch tensor or parameter to a native Python type that is 
        serializable for JSON saving. The function recursively handles both 
        torch.Tensor objects and torch.nn.Parameter objects, ensuring that any tensor 
        values are converted to either scalars (if the tensor is a single value) or 
        lists (if the tensor has multiple elements).

        The method helps prepare the metrics data by converting PyTorch-specific 
        objects to standard Python data types, allowing them to be written to a 
        JSON file.

        Parameters:
            t (Union[torch.Tensor, torch.nn.Parameter, Any]): 
                The input tensor or parameter (or any other type) to convert. 
                It can be a scalar tensor, a multi-element tensor, or any object 
                that doesn't need conversion.

        Returns:
            Union[float, List[float], Any]: 
                The converted value. If the input is a tensor, it will return a 
                float (for scalar tensors) or a list of floats (for multi-element tensors).
                Otherwise, the original value is returned.

        Example:
            tensor = torch.tensor([1.0, 2.0, 3.0])
            converted = self.convert_tensor(tensor)
            # converted will be [1.0, 2.0, 3.0], a list of floats.

        Note:
            - If the input is a `torch.Tensor` with a single element, it will be converted to a scalar float.
            - If the input is a `torch.nn.Parameter`, its data will be recursively converted using `convert_tensor`.
            - The function will return the original value if the input is not a tensor or parameter.
        """

        if isinstance(t, torch.Tensor):
            if t.numel() == 1:
                return t.item()  # Convert scalar tensors to float
            else:
                return t.tolist()  # Convert other tensors to lists
        elif isinstance(t, torch.nn.Parameter):
            return self.convert_tensor(t.data)  # If it's a Parameter, convert its data
        return t  # If it's not a tensor, return as is

    def assert_sparsity(self, sparsity: float) -> None:
        """
        Asserts that the model has reached the expected sparsity level by comparing 
        the actual sparsity of the model’s weight parameters with the target sparsity 
        value, allowing for a small tolerance in the difference. The method counts 
        the total number of parameters and the number of pruned (zeroed-out) parameters 
        and calculates the actual sparsity of the model.

        Parameters:
            sparsity (float): 
                The target sparsity value that the model should have after pruning, 
                represented as a float between 0 and 1 (e.g., 0.9 for 90% sparsity).

        Returns:
            None

        Raises:
            AssertionError: 
                If the model’s actual sparsity differs significantly from the target 
                sparsity (beyond a tolerance of 1e-2), an AssertionError will be raised, 
                indicating a mismatch between the expected and actual sparsity.

        Example:
            # Assume the model has been pruned to a target sparsity of 90%
            pruning.assert_sparsity(0.9)
            # This will verify if the current sparsity of the model is close to 90%.

        Note:
            - The function only considers weight parameters in the model (i.e., parameters 
            containing the substring 'weight' in their names).
            - The method asserts that the sparsity is within a small tolerance (atol=1e-2) 
            to account for minor differences during pruning.
            - After successful verification, the `current_sparsity` attribute is updated 
            with the actual sparsity value of the model.
        """

        total_params_model = 0
        pruned_params_model = 0
        for module in get_pruneable_modules(self.model, self.prunable_layers):
            total_params_model += module.weight.numel()
            pruned_params_model += torch.sum(module.weight == 0).item()

        current_sparsity_model = pruned_params_model / total_params_model

        # Allow for a small tolerance in the sparsity difference
        assert np.isclose(current_sparsity_model, sparsity, atol=1e-2), \
            f"Model sparsity mismatch: {current_sparsity_model} vs {sparsity}"
        
        # Update the current sparsity
        self.current_sparsity = current_sparsity_model

        logger.info(f"Sparsity assertion passed: {current_sparsity_model * 100:.2f}% model.")
