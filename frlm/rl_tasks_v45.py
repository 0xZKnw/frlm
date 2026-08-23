"""Banque de tâches RLVR v4.5 à programmes latents, indépendante des benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from frlm.verifiers_v45 import AnswerSpec, VERIFIER_VERSION


Split = Literal["train", "dev"]
CAPABILITY_WEIGHTS = {
    "reasoning_program": 0.40,
    "grounded": 0.15,
    "constraints": 0.15,
    "code": 0.10,
    "uncertainty": 0.10,
    "state_tracking": 0.10,
}

SCHEMAS_BY_CAPABILITY = {
    "reasoning_program": ("linear_equation", "mean_three", "exact_percentage",
                          "small_power", "signed_product"),
    "grounded": ("grounded_year", "grounded_rooms", "grounded_height"),
    "constraints": ("constraint_number_only", "constraint_json"),
    "code": ("code_0", "code_1", "code_2"),
    "uncertainty": ("contradictory_sources",),
    "state_tracking": ("state_update",),
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


def _reasoning(rng: random.Random, seed: int, split: Split, difficulty: float,
               forced_schema: str | None = None) -> TaskSpec:
    # Ne jamais ajouter ici intervalle/fencepost, reste euclidien, cycle de jours,
    # comparaison de branches, transitivité ou composition multi-opérations : ce
    # sont les familles scellées du benchmark OOD v2.
    schemas = SCHEMAS_BY_CAPABILITY["reasoning_program"]
    schema = forced_schema or rng.choice(schemas)
    if schema not in schemas:
        raise ValueError(f"schéma reasoning inconnu : {schema}")
    scale = 20 + int(180 * difficulty)
    trace: list[str]
    if schema == "linear_equation":
        result, offset = rng.randint(-scale, scale), rng.randint(2, scale)
        total = result + offset
        template, sid = _surface(rng, split, (
            "Trouve l'entier x tel que x + {a} = {b}.",
            "Dans l'égalité ? + {a} = {b}, quelle valeur remplace le point d'interrogation ?",
        ), (
            "Résous cette équation à une inconnue : {b} = {a} + x.",
        ))
        prompt = template.format(a=offset, b=total)
        trace = [f"x = {total} - {offset}", f"x = {result}"]
        answer = AnswerSpec("integer", result)
        program = {"op": "solve_add", "offset": offset, "total": total}
    elif schema == "mean_three":
        result = rng.randint(2, scale)
        delta = rng.randint(1, min(result, max(2, scale // 3)))
        values = (result - delta, result, result + delta)
        template, sid = _surface(rng, split, (
            "Quelle est la moyenne arithmétique de {a}, {b} et {c} ?",
            "Trois mesures valent {a}, {b} et {c}. Donne leur moyenne.",
        ), (
            "Calcule la valeur moyenne du triplet ({a}; {b}; {c}).",
        ))
        prompt = template.format(a=values[0], b=values[1], c=values[2])
        trace = [f"Somme = {sum(values)}", f"{sum(values)} / 3 = {result}"]
        answer = AnswerSpec("integer", result)
        program = {"op": "mean", "args": list(values)}
    elif schema == "exact_percentage":
        rate = rng.choice((10, 20, 25, 50))
        unit = 100 // rate
        base = unit * rng.randint(2, max(3, scale // unit))
        result = base * rate // 100
        template, sid = _surface(rng, split, (
            "Combien représentent {r} % de {n} ?",
            "Calcule exactement {r} pour cent de {n}.",
        ), (
            "Une quantité vaut {n}; quelle part numérique correspond à {r} % ?",
        ))
        prompt = template.format(r=rate, n=base)
        trace = [f"{base} × {rate} / 100 = {result}"]
        answer = AnswerSpec("integer", result)
        program = {"op": "percentage", "base": base, "rate": rate}
    elif schema == "small_power":
        value = rng.randint(2, min(20, 5 + int(15 * difficulty)))
        exponent = rng.choice((2, 3))
        result = value ** exponent
        template, sid = _surface(rng, split, (
            "Combien vaut {n} puissance {p} ?",
            "Calcule {n}^{p}.",
        ), (
            "Évalue la puissance entière de base {n} et d'exposant {p}.",
        ))
        prompt = template.format(n=value, p=exponent)
        trace = [f"{value}^{exponent} = {result}"]
        answer = AnswerSpec("integer", result)
        program = {"op": "pow", "base": value, "exponent": exponent}
    else:
        left = rng.randint(-max(3, scale // 3), max(3, scale // 3))
        right = rng.randint(2, max(3, scale // 4))
        result = left * right
        template, sid = _surface(rng, split, (
            "Calcule le produit signé {a} × {b}.",
            "Quel entier obtient-on en multipliant {a} par {b} ?",
        ), (
            "Effectue cette multiplication avec son signe : ({a}) · ({b}).",
        ))
        prompt = template.format(a=left, b=right)
        trace = [f"{left} × {right} = {result}"]
        answer = AnswerSpec("integer", result)
        program = {"op": "mul", "args": [left, right]}
    prompt += " Réponds par la réponse finale, sans recopier l'énoncé."
    return TaskSpec(_task_id(seed, schema, sid, prompt), schema, sid, split,
                    "reasoning_program", difficulty, prompt, answer, program,
                    tuple(trace), VERIFIER_VERSION, seed, requires_trace=difficulty >= 0.75)


def _grounded(rng: random.Random, seed: int, split: Split, difficulty: float,
              forced_schema: str | None = None) -> TaskSpec:
    town = rng.choice(("Aubeterre", "Belrive", "Clairval", "Montbois", "Rochebrune"))
    year, rooms, height = rng.randint(1840, 2020), rng.randint(3, 30), rng.randint(12, 90)
    context = (f"La station de {town} a ouvert en {year}. Le bâtiment mesure {height} mètres "
               f"et contient {rooms} salles. Sa mission est d'observer le climat local.")
    field = forced_schema.removeprefix("grounded_") if forced_schema else \
        rng.choice(("year", "rooms", "height"))
    if field not in ("year", "rooms", "height"):
        raise ValueError(f"schéma grounded inconnu : {forced_schema}")
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
                    (f"Le champ {field} vaut {value}.",), VERIFIER_VERSION, seed)


def _constraints(rng: random.Random, seed: int, split: Split, difficulty: float,
                 forced_schema: str | None = None) -> TaskSpec:
    a, b = rng.randint(2, 90), rng.randint(2, 90)
    total = a + b
    number_only = (forced_schema == "constraint_number_only" if forced_schema
                   else rng.random() < 0.55)
    if forced_schema not in (None, "constraint_number_only", "constraint_json"):
        raise ValueError(f"schéma constraints inconnu : {forced_schema}")
    if number_only:
        prompt = f"Combien font {a} + {b} ? Réponds uniquement par le nombre."
        answer = AnswerSpec("integer", total, strict_number_only=True)
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
                    VERIFIER_VERSION, seed)


def _uncertainty(rng: random.Random, seed: int, split: Split, difficulty: float,
                 forced_schema: str | None = None) -> TaskSpec:
    if forced_schema not in (None, "contradictory_sources"):
        raise ValueError(f"schéma uncertainty inconnu : {forced_schema}")
    code_a = rng.choice(("AX-12", "BR-7", "CT-44", "DX-9"))
    code_b = rng.choice(tuple(code for code in ("AX-12", "BR-7", "CT-44", "DX-9")
                              if code != code_a))
    prompt = (f"Deux avis officiels publiés au même instant se contredisent : le premier donne "
              f"le code {code_a}, le second le code {code_b}. Aucun avis n'est prioritaire. "
              "Quel code unique doit-on retenir ?")
    sid = f"{split}:{rng.randrange(3 if split == 'train' else 2)}"
    return TaskSpec(_task_id(seed, "contradictory_sources", sid, prompt), "contradictory_sources",
                    sid, split, "uncertainty", difficulty, prompt, AnswerSpec("abstain"),
                    {"op": "unanswerable", "reason": "equal_authority_contradiction"},
                    ("Les deux sources de même autorité se contredisent.",), VERIFIER_VERSION, seed)


def _state(rng: random.Random, seed: int, split: Split, difficulty: float,
           forced_schema: str | None = None) -> TaskSpec:
    if forced_schema not in (None, "state_update"):
        raise ValueError(f"schéma state inconnu : {forced_schema}")
    colors = ("rouge", "bleu", "vert", "jaune")
    first = rng.choice(colors)
    second = rng.choice(tuple(color for color in colors if color != first))
    third = rng.choice(tuple(color for color in colors if color != second))
    prompt = (f"Conversation précédente : utilisateur : l'étiquette du dossier est {first}. "
              f"assistant : noté. utilisateur : remplace-la par {second}. assistant : fait. "
              f"utilisateur : correction finale, mets-la en {third}. Quelle est sa couleur actuelle ?")
    sid = f"{split}:{rng.randrange(3 if split == 'train' else 2)}"
    return TaskSpec(_task_id(seed, "state_update", sid, prompt), "state_update", sid, split,
                    "state_tracking", difficulty, prompt,
                    AnswerSpec("choice", third, choices=colors),
                    {"op": "last_write_wins", "values": [first, second, third]},
                    (f"La dernière mise à jour remplace {second} par {third}.",),
                    VERIFIER_VERSION, seed)


def _code(rng: random.Random, seed: int, split: Split, difficulty: float,
          forced_schema: str | None = None) -> TaskSpec:
    suffix = seed % 1_000_000
    kind = int(forced_schema.removeprefix("code_")) if forced_schema else rng.randrange(3)
    if kind not in (0, 1, 2):
        raise ValueError(f"schéma code inconnu : {forced_schema}")
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
                    {"op": "python_tests", "function": name}, (), VERIFIER_VERSION, seed)


def make_task(seed: int, split: Split = "train", difficulty: float = 0.35,
              capability: str | None = None, schema_id: str | None = None) -> TaskSpec:
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
    return factories[capability](rng, seed, split, difficulty, schema_id)


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
