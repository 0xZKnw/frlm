"""Variante locale Qwen4-exp dense, exportable avec le code HF associé.

Seuls les blocs MoE deviennent des SwiGLU ; mélangeurs, GR4, PLE et cache
restent ceux de Transformers 5.17.0. Aucun poids pré-entraîné n'est chargé.
"""
from copy import copy

from huggingface_hub.dataclasses import strict
from transformers import Qwen4ExpTextConfig, Qwen4ExpForCausalLM
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextMLP


@strict
class Qwen4ExpDenseConfig(Qwen4ExpTextConfig):
    model_type = "frlm_qwen4exp_dense"
    intermediate_size: int = 3840
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 1
    shared_expert_intermediate_size: int = 1
    router_aux_loss_coef: float = 0.0
    output_router_logits: bool = False

    def validate_architecture(self):
        if self.num_experts != 0 or self.num_experts_per_tok != 0:
            raise ValueError("Qwen4-exp dense exige zéro expert et zéro sélection")
        if self.router_aux_loss_coef != 0 or self.output_router_logits:
            raise ValueError("Qwen4-exp dense ne possède pas de routeur")
        if self.intermediate_size <= 0:
            raise ValueError("SwiGLU dense : intermediate_size doit être positif")
        # L'amont impose des experts. Valider ses autres invariants sur une copie
        # de configuration ; le modèle réel garde num_experts=0.
        common = copy(self)
        common.num_experts = common.num_experts_per_tok = 1
        Qwen4ExpTextConfig.validate_architecture(common)


class Qwen4ExpDenseForCausalLM(Qwen4ExpForCausalLM):
    config_class = Qwen4ExpDenseConfig

    def __init__(self, config):
        super().__init__(config)
        # Les tenseurs d'experts amont sont vides (num_experts=0). Remplacer tout
        # le bloc, y compris gate et shared_expert, avant le premier forward.
        for layer in self.model.layers:
            layer.mlp = Qwen4ExpTextMLP(config, config.intermediate_size)
            layer.mlp.apply(self._initialize_weights)


Qwen4ExpDenseConfig.register_for_auto_class()
Qwen4ExpDenseForCausalLM.register_for_auto_class("AutoModelForCausalLM")
