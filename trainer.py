from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch

# abstract class
class Trainer:
    def __init__(self) -> None:
        pass
    
    def train(self) -> None:
    
    def compute_loss(
        self,
        model: Union[nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]: