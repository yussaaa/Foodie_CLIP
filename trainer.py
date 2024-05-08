from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import nn
import torch
from transformer import Trainer, TrainingArguments, PreTrainedModel, DataCollator, PreTrainedTokenizerBase
from datasets import Dataset

# abstract class
class CLIPTrainer(Trainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        training_args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        Loss_FN: Optional[Union[nn.Module]] = None
    ):
        pass
        
    
    def compute_loss(
        self,
        model: Union[nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        text_input = inputs["text_input"]
        image_input = inputs["image_input"]
        similarity_logits = model(text_input=text_input, image_input=image_input)
        loss = self.loss_fn(similarity_logits)
        return loss