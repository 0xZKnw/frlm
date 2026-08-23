"""Banque de tâches RLVR v4.5 à programmes latents, indépendante des benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from frlm.verifiers_v45 import AnswerSpec


Split = Literal["train", "dev"]
CAPABILITY_WEIGHTS = {
    "reasoning_program": 0.40,
    "grounded": 0.15,
    "constraints": 0.15,
    "code": 0.10,
    "uncertainty": 0.10,
    "state_tracking": 0.10,
}
NAMES = ("Aïcha", "Bastien", "Chloé", "Dina", "Émile", "Fatou", "Gabriel", "Hana")
OBJECTS = ("pommes", "jetons", "cartes", "cahiers", "perles", "tickets")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    schema_id: str
    surface_id: str
    split: Split
    capability: str
    difficulty: float
    prompt: str
    answer: AnswerSpec
    latent_program: dict
    canonical_trace: tuple[str, ...]
    verifier_version: str
    seed: int
    requires_trace: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["answer"] = self.answer.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "TaskSpec":
        data = dict(payload)
        data["answer"] = AnswerSpec.from_dict(data["answer"])
        data["canonical_trace"] = tuple(data.get("canonical_trace") or ())
        return cls(**data)


def _task_id(seed: int, schema: str, surface: str, prompt: str) -> str:
    raw = f"v45|{seed}|{schema}|{surface}|{prompt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _surface(rng: random.Random, split: Split, train: tuple[str, ...], dev: tuple[str, ...]):
    choices = train if split == "train" else dev
    index = rng.randrange(len(choices))
    return choices[index], f"{split}:{index}"


def _reasoning(rng: random.Random, seed: int, split: Split, difficulty: float) -> TaskSpec:
    schemas = ("stock_three_ops", "inclusive_count", "remainder", "weekly_cycle", "compare_totals")
    schema = rng.choice(schemas)
    scale = 20 + int(180 * difficulty)
    trace: list[str]
    if schema == "stock_three_ops":
        start, packs, size = rng.randint(3, scale), rng.randint(2, 8), rng.randint(2, 12)
        used = rng.randint(1, min(start + packs * size - 1, scale))
        result = start + packs * size - used
        name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
        template, sid = _surface(rng, split, (
            "{n} possède {a} {o}, reçoit {b} paquets de {c}, puis en utilise {d}. Combien lui en reste-t-il ?",
            "Stock initial {a}; ajoute {b} lots de {c}, puis retire {d}. Donne le stock final.",
            "Calcule {a} + ({b} × {c}) - {d}.",
        ), (
            "Une réserve contient {a} {o}. {b} caisses de {c} arrivent et {d} unités partent. Quel solde reste ?",
            "Après une hausse de {b} groupes de {c} et une baisse de {d}, quel devient le niveau {a} ?",
        ))
        prompt = template.format(n=name, o=obj, a=start, b=packs, c=size, d=used)
        trace = [f"{packs} × {size} = {packs * size}",
                 f"{start} + {packs * size} - {used} = {result}"]
        answer = AnswerSpec("integer", result)
        program = {"op": "sub", "args": [{"op": "add", "args": [start, {"op": "mul", "args": [packs, size]}]}, used]}
    elif schema == "inclusive_count":
        first = rng.randint(0, scale)
        last = first + rng.randint(2, max(3, scale // 2))
        result = last - first + 1
        template, sid = _surface(rng, split, (
            "Combien d'entiers y a-t-il de {a} à {b}, bornes incluses ?",
            "Les dossiers sont numérotés de {a} à {b}. Combien de numéros différents ?",
        ), (
            "Une série commence au rang {a} et finit au rang {b}; les extrémités comptent. Quelle est sa longueur ?",
        ))
        prompt = template.format(a=first, b=last)
        trace = [f"{last} - {first} + 1 = {result}"]
        answer = AnswerSpec("integer", result)
        program = {"op": "inclusive_count", "first": first, "last": last}
    elif schema == "remainder":
        divisor = rng.randint(3, 12)
        quotient = rng.randint(2, max(3, scale // divisor))
        remainder = rng.randint(1, divisor - 1)
        total = divisor * quotient + remainder
        template, sid = _surface(rng, split, (
            "On range {n} objets par groupes de {d}. Combien restent hors des groupes complets ?",
            "Quel est le reste de la division de {n} par {d} ?",
        ), (
            "Des cartons prennent {d} unités. Après avoir rempli les cartons avec {n} unités, quel est le reliquat ?",
        ))
        prompt = template.format(n=total, d=divisor)
        trace = [f"{total} = {quotient} × {divisor} + {remainder}"]
        answer = AnswerSpec("integer", remainder)
        program = {"op": "mod", "args": [total, divisor]}
    elif schema == "weekly_cycle":
        days = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
        start, offset = rng.randrange(7), rng.randint(2, 10 + int(70 * difficulty))
        result = days[(start + offset) % 7]
        template, sid = _surface(rng, split, (
            "Nous sommes {day}. Quel jour serons-nous dans {n} jours ?",
            "Avance de {n} jours à partir de {day}.",
        ), (
            "Le calendrier indique {day}. Après {n} journées complètes, quel jour affichera-t-il ?",
        ))
        prompt = template.format(day=days[start], n=offset)
        trace = [f"{offset} modulo 7 = {offset % 7}", f"Jour final : {result}"]
        answer = AnswerSpec("choice", result, choices=days)
        program = {"op": "cycle", "size": 7, "start": start, "offset": offset}
    else:
        left_a, left_b = rng.randint(2, scale), rng.randint(2, scale)
        right_a, right_b = rng.randint(2, scale), rng.randint(2, scale)
        while left_a + left_b == right_a + right_b:
            right_b += 1
        result = "option A" if left_a + left_b > right_a + right_b else "option B"
        template, sid = _surface(rng, split, (
            "Option A vaut {a}+{b}; option B vaut {c}+{d}. Laquelle donne le plus grand total ?",
            "Compare les sommes A={a}+{b} et B={c}+{d}. Quel choix gagne ?",
        ), (
            "Deux équipes marquent A: {a} puis {b}, B: {c} puis {d}. Qui obtient le total maximal ?",
        ))
        prompt = template.format(a=left_a, b=left_b, c=right_a, d=right_b)
        trace = [f"A = {left_a + left_b}", f"B = {right_a + right_b}", result]
        answer = AnswerSpec("choice", result, aliases=(result[-1],), choices=("option A", "option B", "A", "B"))
        program = {"op": "argmax", "a": [left_a, left_b], "b": [right_a, right_b]}
    prompt += " Réponds par la réponse finale, sans recopier l'énoncé."
    return TaskSpec(_task_id(seed, schema, sid, prompt), schema, sid, split,
                    "reasoning_program", difficulty, prompt, answer, program,
                    tuple(trace), "typed_v45_1", seed, requires_trace=difficulty >= 0.75)


def _grounded(rng: random.Random, seed: int, split: Split, difficulty: float) -> TaskSpec:
    town = rng.choice(("Aubeterre", "Belrive", "Clairval", "Montbois", "Rochebrune"))
    year, rooms, height = rng.randint(1840, 2020), rng.randint(3, 30), rng.randint(12, 90)
    context = (f"La station de {town} a ouvert en {year}. Le bâtiment mesure {height} mètres "
               f"et contient {rooms} salles. Sa mission est d'observer le climat local.")
    field = rng.choice(("year", "rooms", "height"))
    question = {"year": "En quelle année a-t-elle ouvert ?",
                "rooms": "Combien contient-elle de salles ?",
                "height": "Quelle est sa hauteur en mètres ?"}[field]
    value = {"year": year, "rooms": rooms, "height": height}[field]
    template, sid = _surface(rng, split, (
        "Contexte : {ctx}\nQuestion : {q}",
        "Lis cette notice puis réponds brièvement.\n{ctx}\n{q}",
    ), ("Appuie-toi exclusivement sur le document suivant.\n{ctx}\nDemande : {q}",))
    prompt = template.format(ctx=context, q=question)
    schema = f"grounded_{field}"
    return TaskSpec(_task_id(seed, schema, sid, prompt), schema, sid, split, "grounded",
                    difficulty, prompt, AnswerSpec("integer", value),
                    {"op": "extract", "field": field, "context_sha256": hashlib.sha256(context.encode()).hexdigest()},
                    (f"Le champ {field} vaut {value}.",), "typed_v45_1", seed)


def _constraints(rng: random.Random, seed: int, split: Split, difficulty: float) -> TaskSpec:
    a, b = rng.randint(2, 90), rng.randint(2, 90)
    total = a + b
    if rng.random() < 0.55:
        prompt = f"Combien font {a} + {b} ? Réponds uniquement par le nombre."
        answer = AnswerSpec("integer", total)
        schema, program = "constraint_number_only", {"op": "add", "args": [a, b], "format": "number"}
    else:
        value = {"operation": "addition", "resultat": total}
        prompt = (f"Calcule {a} + {b}. Réponds uniquement avec un objet JSON ayant exactement "
                  "les clés operation (texte) et resultat (entier).")
        answer = AnswerSpec("json", value, json_schema={"operation": "string", "resultat": "integer"})
        schema, program = "constraint_json", {"op": "add", "args": [a, b], "format": "json"}
    sid = f"{split}:{rng.randrange(3 if split == 'train' else 2)}"
    return TaskSpec(_task_id(seed, schema, sid, prompt), schema, sid, split, "constraints",
                    difficulty, prompt, answer, program, (f"{a} + {b} = {total}",),
                    "typed_v45_1", seed)


def _uncertainty(rng: random.Random, seed: int, split: Split, difficulty: float) -> TaskSpec:
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    known = rng.randint(4, 80)
    prompt = (f"Le texte indique que {name} possède {known} {obj}. "
              "Combien de portes possède sa maison ?")
    sid = f"{split}:{rng.randrange(3 if split == 'train' else 2)}"
    return TaskSpec(_task_id(seed, "missing_information", sid, prompt), "missing_information",
                    sid, split, "uncertainty", difficulty, prompt, AnswerSpec("abstain"),
                    {"op": "unanswerable", "missing": "door_count"},
                    ("Le nombre de portes n'est pas fourni.",), "typed_v45_1", seed)


def _state(rng: random.Random, seed: int, split: Split, difficulty: float) -> TaskSpec:
    start, gain, loss = rng.randint(3, 60), rng.randint(2, 30), rng.randint(1, 20)
    result = start + gain - loss
    prompt = (f"Conversation précédente : utilisateur : le compteur vaut {start}. assistant : compris. "
              f"utilisateur : ajoute {gain}. assistant : il vaut {start + gain}. "
              f"utilisateur : retire maintenant {loss}. Quelle est sa nouvelle valeur ?")
    sid = f"{split}:{rng.randrange(3 if split == 'train' else 2)}"
    return TaskSpec(_task_id(seed, "state_update", sid, prompt), "state_update", sid, split,
                    "state_tracking", difficulty, prompt, AnswerSpec("integer", result),
                    {"op": "sub", "args": [{"op": "add", "args": [start, gain]}, loss]},
                    (f"{start} + {gain} = {start + gain}", f"{start + gain} - {loss} = {result}"),
                    "typed_v45_1", seed)


def _code(rng: random.Random, seed: int, split: Split, difficulty: float) -> TaskSpec:
    suffix = seed % 1_000_000
    kind = rng.randrange(3)
    if kind == 0:
        name = f"double_{suffix}"
        prompt = f"Écris uniquement le code Python d'une fonction `{name}(n)` qui renvoie le double de n."
        tests = (([3], 6), ([-4], -8), ([0], 0))
    elif kind == 1:
        name = f"borne_{suffix}"
        prompt = (f"Écris uniquement le code Python de `{name}(n, minimum, maximum)` qui borne n "
                  "entre minimum et maximum.")
        tests = (([12, 0, 10], 10), ([3, 0, 10], 3), ([-2, 0, 10], 0))
    else:
        name = f"somme_carres_{suffix}"
        prompt = f"Écris uniquement le code Python de `{name}(valeurs)` qui renvoie la somme des carrés."
        tests = (([[1, 2, 3]], 14), ([[]], 0), ([[-2, 3]], 13))
    sid = f"{split}:{kind}"
    answer = AnswerSpec("code", function_name=name, tests=tests)
    return TaskSpec(_task_id(seed, f"code_{kind}", sid, prompt), f"code_{kind}", sid,
                    split, "code", difficulty, prompt, answer,
                    {"op": "python_tests", "function": name}, (), "typed_v45_1", seed)


def make_task(seed: int, split: Split = "train", difficulty: float = 0.35,
              capability: str | None = None) -> TaskSpec:
    if split not in ("train", "dev"):
        raise ValueError("les tâches sealed vivent hors du générateur de train")
    difficulty = max(0.0, min(1.0, float(difficulty)))
    rng = random.Random(seed * 1_000_003 + (0 if split == "train" else 97_531))
    if capability is None:
        names, weights = zip(*CAPABILITY_WEIGHTS.items())
        capability = rng.choices(names, weights=weights, k=1)[0]
    factories = {
        "reasoning_program": _reasoning, "grounded": _grounded,
        "constraints": _constraints, "code": _code,
        "uncertainty": _uncertainty, "state_tracking": _state,
    }
    if capability not in factories:
        raise ValueError(f"capacité RL inconnue : {capability}")
    return factories[capability](rng, seed, split, difficulty)


def write_bank(path: Path, count: int, split: Split, seed: int = 451_900) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    hashes = []
    counts: dict[str, int] = {}
    with tmp.open("w", encoding="utf-8") as stream:
        for index in range(count):
            task = make_task(seed + index, split=split, difficulty=(index % 11) / 10)
            line = json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True)
            stream.write(line + "\n")
            hashes.append(hashlib.sha256(line.encode("utf-8")).hexdigest())
            counts[task.capability] = counts.get(task.capability, 0) + 1
    tmp.replace(path)
    digest = hashlib.sha256("".join(hashes).encode()).hexdigest()
    manifest = {"schema": "frlm-rl-tasks-v45-1", "path": str(path), "count": count,
                "split": split, "seed": seed, "sha256": digest, "capabilities": counts}
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def iter_bank(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield TaskSpec.from_dict(json.loads(line))


def main():
    parser = argparse.ArgumentParser(description="Génère une banque RLVR v4.5 non scellée")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--seed", type=int, default=451_900)
    args = parser.parse_args()
    print(json.dumps(write_bank(args.out, args.count, args.split, args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

