
from transformers.models.mistral.modeling_mistral import MistralConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
class LoraMoeConfig(Qwen2VLConfig):

    def __init__(
        self,
        experts_rank=8,
        experts_scale=2.0,
        num_experts_per_tok=1,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        **kwargs
    ):

        self.experts_rank = experts_rank
        self.experts_scale = experts_scale


        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        super().__init__(**kwargs)
