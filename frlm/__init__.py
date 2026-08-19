"""frlm — petit LLM français de raisonnement (58M), entraîné sur une RTX 4060.

Modules :
    model   l'architecture type Qwen3.5 (attention gated, GQA, RoPE partiel)
    data    corpus FR, tokenizer BPE à chiffres séparés, binarisation
    synth   générateur de problèmes maths/logique FR (solutions calculées)
    optim   Muon + schedules de LR
    rl      GRPO : renforcement à récompenses vérifiables

Point d'entrée : run.py à la racine du dépôt.
"""
