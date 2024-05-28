from foodie_clip.foodieclip import FoodieClip, ClIPLoss
from trainer.utils import CLIPDataCollator, CLIPDataset
from trainer.evaluate import compute_accuracy
from foodie_clip.foodieclip import FoodieClip, ClIPLoss
from trainer.utils import CLIPDataset, CLIPDataCollator
from trainer.clip_trainer import CLIPTrainer
from transformers import TrainingArguments, AutoProcessor, CLIPConfig

# test trainer
if __name__ == '__main__':
    # loading model
    foodieclip_config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32") 
    foodieclip = FoodieClip(foodieclip_config, projection_dim=64)
    # loading dataset
    train_dataset = CLIPDataset(train=True, class_max_datapoints=200)
    eval_dataset = CLIPDataset(train=False, class_max_datapoints=100)
    # set data collator
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    data_collator = CLIPDataCollator(processor=processor)
    # set up training args
    training_args = TrainingArguments(
        output_dir = "output/",
        per_device_train_batch_size=25,
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
        compute_metrics=compute_accuracy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        Loss_FN=ClIPLoss
        )
    trainer.train()