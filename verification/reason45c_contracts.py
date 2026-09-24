"""Recherche symbolique sur le code de production, sans I/O ni entraînement.

CrossHair explore les assertions après les préconditions ; un résultat inconclusif
ou l'absence de contre-exemple n'est pas une preuve du pipeline complet.
"""
from frlm.reason_bootstrap_v45 import _natural, evaluate_ast
from frlm.data import token_sampling_weight
from frlm.modal_preflight import with_gpu_peak


class Infix:
    def choice(self, templates):
        return templates[-1]


def subtraction_preserves_grouping(a: int, b: int, c: int):
    assert -600 <= a <= 600
    assert -600 <= b <= 600
    assert -600 <= c <= 600
    assert c != 0
    const = lambda n: {"op": "const", "value": n}
    sub = lambda x, y: {"op": "sub", "args": [x, y]}
    left = sub(sub(const(a), const(b)), const(c))
    right = sub(const(a), sub(const(b), const(c)))
    assert evaluate_ast(left) == a - b - c
    assert evaluate_ast(right) == a - b + c
    assert _natural(left, "train", Infix()) != _natural(right, "train", Infix())


def equal_token_mix(short: int, long: int):
    assert 1 <= short <= 512
    assert 1 <= long <= 512
    a = token_sampling_weight(0.5, 1, short)
    b = token_sampling_weight(0.5, 1, long)
    # À taille finie d'update, seules les contributions en espérance sont visées.
    assert abs(a * short - b * long) < 1e-12


def gpu_peak_only_training(peak: int):
    assert 1 <= peak <= 2000
    profile = "python -m frlm.eval_reason_bootstrap_v45 --stage sft --run tiny"
    assert with_gpu_peak(profile, peak) == profile
    for command in ("python run.py train", "python run.py mid", "python run.py sft",
                    "python -m frlm.bench_speed"):
        result = with_gpu_peak(command, peak)
        assert result == command + " --gpu-peak-tflops " + str(peak)
        assert with_gpu_peak(result, peak) == result
