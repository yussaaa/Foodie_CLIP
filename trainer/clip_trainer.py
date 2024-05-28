from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import nn
import torch
from transformers import Trainer, TrainingArguments, PreTrainedModel, DataCollator, AutoProcessor, CLIPConfig
from importlib.util import find_spec
from torch.utils.data import Dataset
from .evaluate import compute_accuracy

def is_peft_available():
    return find_spec("peft") is not None

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    
class CLIPTrainer(Trainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        training_args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        compute_metrics=compute_accuracy,
        processor: Optional[AutoProcessor] = None,
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        Loss_FN: Optional[Union[nn.Module]] = None
    ):
        # TO DO: support peft
        # # wrapped model into a peft model
        
        # check max_length arg
        if (not hasattr(training_args , "max_length") or training_args.max_length is None) and max_length is None:
            max_length = 512
        elif max_length is None:
            max_length = training_args.max_length
        max_length = min(max_length, model.config.text_config.max_position_embeddings)
        
        # check processor arg
        if processor is None:
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.processor = processor
        
        # check data_collator arg
        if data_collator is None:
            # default data_collator class is CLIPDataCollator
            self.data_collator = CLIPDataCollator(processor=self.processor, max_length=max_length)
        else:
            self.data_collator = data_collator
        self.data_collator.set_max_length(max_length)

        self.loss_fn = Loss_FN()
        
        super().__init__(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers
        )
    
    def compute_loss(
        self,
        model: Union[nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_logits=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        text_input = inputs["texts_input"]
        image_input = inputs["images_input"]
        mask = inputs["mask"]
        similarity_logits = model(text_input=text_input, image_input=image_input)
        loss = self.loss_fn(similarity_logits, mask)
        if not return_logits:
            return loss
        else:
            return loss, similarity_logits
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, similarity_logits = self.compute_loss(model=model, inputs=inputs, return_logits=True)
            loss = loss.detach()
            similarity_logits = similarity_logits.detach()
        return loss, similarity_logits, inputs["mask"]
        
# test trainer
if __name__ == '__main__':
    import os
    path = os.getcwd()
    import sys
    # append the path of the parent directory
    sys.path.append(path)
    from foodie_clip.foodieclip import FoodieClip, ClIPLoss
    from utils import CLIPDataset, CLIPDataCollator
    # loading model
    foodieclip_config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32") 
    foodieclip = FoodieClip(foodieclip_config, projection_dim=64)
    # loading dataset
    train_dataset = CLIPDataset(train=True, class_max_datapoints=2)
    eval_dataset = CLIPDataset(train=False, class_max_datapoints=2)
    # set data collator
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    data_collator = CLIPDataCollator(processor=processor)
    # set up training args
    training_args = TrainingArguments(
        output_dir = "output/",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        logging_steps=1,
        num_train_epochs = 2,
        report_to=None,
        eval_steps = 5,
        fp16=False,
        remove_unused_columns=False
    )
    
    trainer = CLIPTrainer(
        model=foodieclip,
        training_args = training_args,
        data_collator=data_collator,
        max_length=128,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        Loss_FN=ClIPLoss
        )
    trainer.train()