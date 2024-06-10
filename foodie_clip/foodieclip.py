import torch.nn as nn
from transformers import CLIPModel, CLIPConfig
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import requests
from transformers import (
    CLIPModel,
    CLIPTextModel,
    CLIPPreTrainedModel,
    PreTrainedModel,
    CLIPVisionModel,
    CLIPConfig,
    AutoProcessor,
)
import torch.nn.functional as F


class ClIPLoss(nn.Module):
    def __init__(self):
        super(ClIPLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def contrastive_loss(self, similarity_logits, mask):
        target = self.softmax(mask)
        return F.cross_entropy(input=similarity_logits, target=target)

    def forward(self, similarity_logits, mask):
        # print(similarity_logits.shape, mask.shape)
        image_logits_loss = self.contrastive_loss(similarity_logits, mask)
        text_logits_loss = self.contrastive_loss(similarity_logits.t(), mask.t())
        return (image_logits_loss + text_logits_loss) / 2


class FoodieClipSimilarity(PreTrainedModel, nn.Module):
    name = "FoodieClipSimilarity"
    config_class = CLIPConfig

    def __init__(self, config):
        super(FoodieClipSimilarity, self).__init__(config)
        self.config = config
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )
        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def get_similarity(
        self,
        image_embedding: Optional[torch.FloatTensor] = None,
        text__embedding: Optional[torch.FloatTensor] = None,
    ) -> Optional[torch.FloatTensor]:
        # normalized features
        image_embedding = image_embedding / image_embedding.norm(
            p=2, dim=-1, keepdim=True
        )
        text__embedding = text__embedding / text__embedding.norm(
            p=2, dim=-1, keepdim=True
        )
        # cosine similarity
        logit_scale = self.logit_scale.exp()
        image_logits = torch.matmul(image_embedding, text__embedding.t()) * logit_scale
        return image_logits

    # retrieve process, return the topk matching candidates
    @torch.no_grad()
    def forward(
        self, text_input, image_input, top_k=1, retrieve="text"
    ) -> Optional[List[List]]:
        image_embedding = image_input["embedding"]
        text_embedding = text_input["embedding"]
        image_logits = self.get_similarity(
            image_embedding=image_embedding, text_embedding=text_embedding
        )
        # get the top k best match texts for each input image
        if retrieve == "text":
            top_k_values, top_k_indices = (
                torch.topk(image_logits, top_k, dim=-1).indices.tolist(),
            )
        else:
            top_k_values, top_k_indices = torch.topk(
                image_logits.t(), top_k, dim=-1
            ).indices.tolist()
        return top_k_values, top_k_indices


class FoodieClipPreTrain(CLIPPreTrainedModel):
    name = "FoodieClipPreTrain"
    backbone_class = None

    def __init__(self, config):
        super(FoodieClipPreTrain, self).__init__(config)
        if type(config) == str:
            return self.__class__.backbone_class.from_pretrained(config)
        else:
            return self.__class__.backbone_class(config)

    def save_pretrained(self, MODEL_PATH):
        self.config._name_or_path = MODEL_PATH
        super().save_pretrained(self.config._name_or_path)


class FoodieClipTextModel(FoodieClipPreTrain):
    name = "FoodieClipTextModel"
    backbone_class = CLIPTextModel

    def __init__(self, config):
        backbone = super(FoodieClipTextModel, self).__init__(config.text_config)
        self.config = config
        self.text_config = self.config.text_config
        # self.text_hidden_size = backbone.config.hidden_size
        self.text_model = backbone.text_model
        self.text_projection = nn.Linear(
            self.text_config.hidden_size, self.config.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_text_feature(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = False,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> Optional[Tuple]:
        input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        text_output = self.text_model(**input)
        # the final hidden state output of the input[:], shape(batch, seq_len, hidden_size)
        last_hidden_state = text_output.last_hidden_state
        # take the first EOS token hidden state for embedding, shape(batch, hidden_size)
        pool_hidden_state = text_output.pooler_output
        # get the text embedding from the choosen hidden state
        text_embedding = self.text_projection(pool_hidden_state)
        return text_embedding, pool_hidden_state

    # get the text feature
    # use for initializing or update the embedding vectors db
    def forward(self, text_input, required_grad=False):
        if required_grad:
            text_embedding, pool_hidden_state = self.get_text_feature(**text_input)
        else:
            # no gradient calculation
            with torch.no_grad():
                text_embedding, pool_hidden_state = self.get_text_feature(**text_input)
        return text_embedding.cpu().detach()


class FoodieClipImageModel(FoodieClipPreTrain):
    name = "FoodieClipImageModel"
    backbone_class = CLIPVisionModel

    def __init__(self, config):
        backbone = super(FoodieClipImageModel, self).__init__(config.vision_config)
        self.config = config
        self.vision_config = self.config.vision_config
        self.vision_model = backbone.vision_model
        # self.text_hidden_size = backbone.config.hidden_size
        self.visual_projection = nn.Linear(
            self.vision_config.hidden_size, self.config.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_image_feature(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Optional[Tuple]:
        """input
        pixel_values: the input images
        return_dict: if return_dict is None, return the last hidden state and the pooled hidden state
        """
        """ return
        pool_hidden_state: the first EOS token hidden state, shape(batch, hidden_state_dim)
        image_embedding: the embedding of the input images, for calculating the similarity, shape(batch, embedding_dim)
        """
        input = {
            "pixel_values": pixel_values,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        vision_output = self.vision_model(**input)
        # the final hidden state output of the input[:], shape(batch, seq_len, hidden_size)
        last_hidden_state = vision_output.last_hidden_state
        # take the first EOS token hidden state for embedding, shape(batch, hidden_size)
        pool_hidden_state = vision_output.pooler_output
        # get the image embedding from the choosen hidden state
        image_embedding = self.visual_projection(pool_hidden_state)
        return image_embedding, pool_hidden_state

    # get image input
    def forward(self, image_input, required_grad=False):
        if required_grad:
            image_embedding, pool_hidden_state = self.get_image_feature(**image_input)
        else:
            # no gradient calculation
            with torch.no_grad():
                image_embedding, pool_hidden_state = self.get_image_feature(
                    **image_input
                )
        return image_embedding.cpu().detach()


class FoodieClip(FoodieClipPreTrain):
    name = "FoodieClip"
    backbone_class = CLIPModel

    def __init__(self, config, projection_dim=None):
        backbone = super(FoodieClip, self).__init__(config)
        self.text_model = backbone.text_model
        self.vision_model = backbone.vision_model
        self.config = config
        self.logit_scale = backbone.logit_scale
        # reinitialize the projection layer
        if projection_dim:
            self.config.projection_dim = projection_dim
            self.text_projection = nn.Linear(
                self.config.text_config.hidden_size,
                self.config.projection_dim,
                bias=False,
            )
            self.visual_projection = nn.Linear(
                self.config.vision_config.hidden_size,
                self.config.projection_dim,
                bias=False,
            )
        else:
            self.text_projection = backbone.text_projection
            self.visual_projection = backbone.visual_projection
        # Initialize weights and apply final processing
        self.post_init()

    def get_image_feature(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Optional[Tuple]:
        """input
        pixel_values: the input images
        return_dict: if return_dict is None, return the last hidden state and the pooled hidden state
        """
        """ return
        pool_hidden_state: the first EOS token hidden state, shape(batch, hidden_state_dim)
        image_embedding: the embedding of the input images, for calculating the similarity, shape(batch, embedding_dim)
        """
        input = {
            "pixel_values": pixel_values,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        vision_output = self.vision_model(**input)
        # the final hidden state output of the input[:], shape(batch, seq_len, hidden_size)
        last_hidden_state = vision_output.last_hidden_state
        # take the first EOS token hidden state for embedding, shape(batch, hidden_size)
        pool_hidden_state = vision_output.pooler_output
        # get the image embedding from the choosen hidden state
        image_embedding = self.visual_projection(pool_hidden_state)
        return image_embedding, pool_hidden_state

    def get_text_feature(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = False,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> Optional[Tuple]:
        input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        text_output = self.text_model(**input)
        # the final hidden state output of the input[:], shape(batch, seq_len, hidden_size)
        last_hidden_state = text_output.last_hidden_state
        # take the first EOS token hidden state for embedding, shape(batch, hidden_size)
        pool_hidden_state = text_output.pooler_output
        # get the text embedding from the choosen hidden state
        text_embedding = self.text_projection(pool_hidden_state)
        return text_embedding, pool_hidden_state

    def get_similarity(
        self,
        image_embedding: Optional[torch.FloatTensor] = None,
        text_embedding: Optional[torch.FloatTensor] = None,
    ) -> Optional[torch.FloatTensor]:
        # normalized features
        image_embedding = image_embedding / image_embedding.norm(
            p=2, dim=-1, keepdim=True
        )
        text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity
        logit_scale = self.logit_scale.exp()
        image_logits = torch.matmul(image_embedding, text_embedding.t()) * logit_scale
        # print(image_embedding.shape, text_embedding.shape, image_logits.shape)
        return image_logits

    def forward(self, text_input, image_input):
        text_embedding, pool_hidden_state = self.get_text_feature(**text_input)
        image_embedding, pool_hidden_state = self.get_image_feature(**image_input)
        similarity = self.get_similarity(
            image_embedding=image_embedding, text_embedding=text_embedding
        )
        return similarity


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image_input = processor(images=image, return_tensors="pt", padding=True)
    text_input = processor(
        text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True
    )
    """
    test for FoodieClip
    """
    # loading a pretrain model from hugging face
    foodieclip_config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
    foodieclip = FoodieClip(foodieclip_config, projection_dim=64)
    # save a finetune model to /MODEL_PATH
    foodieclip.save_pretrained("../model")
    # loading a finetune model from /MODEL_PATH
    foodieclip = FoodieClip.from_pretrained("../model")
    foodieclip_image_embedding = foodieclip.get_image_feature(**image_input)[0]
    foodieclip_text_embedding = foodieclip.get_text_feature(**text_input)[0]
    print(
        f"foodie model output {foodieclip(text_input=text_input, image_input=image_input)}"
    )
    """
    test for FoodieClipImageModel
    """
    # loading FoodieClipImageModel from path
    foodie_image_model = FoodieClipImageModel.from_pretrained("../model")
    foodie_image_model_output = foodie_image_model(image_input)
    """
    test for FoodieClipTextModel
    """
    foodie_text_model = FoodieClipTextModel.from_pretrained("../model")
    foodie_text_model_output = foodie_text_model(text_input)
    """
    test for output consistency
    """
    # check the text embedding consistency between foodieclip and foodiecliptext
    print(
        f"text embedding consistency : {foodieclip_text_embedding == foodie_text_model_output}"
    )
    # check the image embedding consistency between foodieclip and foodieclipimage
    print(
        f"image embedding consistency : {foodieclip_image_embedding == foodie_image_model_output}"
    )
