"""Vérificateurs typés pour le RLVR local de FRLM v4.5.

La récompense primaire ne doit jamais dépendre d'une inclusion de sous-chaîne
approximative. Ce module sépare donc extraction, validation de type et score de
format, et renvoie un code d'échec auditable pour chaque génération.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Literal

from frlm.data import EOT, IM_END, THINK, THINK_END


AnswerKind = Literal["integer", "rational", "choice", "entity", "json", "abstain", "code"]


@dataclass(frozen=True)
class AnswerSpec:
    kind: AnswerKind
    value: Any = None
    aliases: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()
    json_schema: dict[str, str] | None = None
    tests: tuple[tuple[list[Any], Any], ...] = ()
    function_name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "AnswerSpec":
        data = dict(payload)
        for key in ("aliases", "choices"):
            data[key] = tuple(data.get(key) or ())
        data["tests"] = tuple((list(args), expected) for args, expected in data.get("tests") or ())
        return cls(**data)


@dataclass(frozen=True)
class Verification:
    primary_success: bool
    primary_score: float
    process_score: float = 0.0
    format_score: float = 0.0
    parsed_answer: Any = None
    failure_code: str | None = None
    details: dict = field(default_factory=dict)


_NUMBER = r"[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*[-+]?\d+)?"
_NUMBER_RE = re.compile(rf"(?<![\w])({_NUMBER})(?![\w])")
_MARKED_NUMBER_RE = re.compile(
    rf"(?:réponse|résultat|final|donc)\s*(?:est|vaut|:|=)?\s*({_NUMBER})(?![\w])",
    re.IGNORECASE,
)
_ABSTAIN_RE = re.compile(
    r"\b(?:impossible (?:de|à)|ne (?:peut|peux) pas (?:savoir|déterminer)|"
    r"pas (?:précisé|indiqué|fourni)|information(?:s)? (?:insuffisante|manquante)s?|"
    r"le (?:texte|contexte|énoncé) ne (?:dit|permet) pas|indéterminable|"
    r"les informations se contredisent|aucune information)\b",
    re.IGNORECASE,
)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def final_text(generation: str) -> str:
    """Extrait uniquement la réponse visible après le dernier bloc de travail."""
    text = generation.replace(IM_END, "\n").replace(EOT, "\n")
    if THINK_END in text:
        text = text.rsplit(THINK_END, 1)[-1]
    elif THINK in text:
        text = text.split(THINK, 1)[0]
    return text.strip()


def _fraction(raw: str) -> Fraction | None:
    raw = raw.strip().replace(" ", "").replace(",", ".")
    try:
        if "/" in raw:
            left, right = raw.split("/", 1)
            return Fraction(left) / Fraction(right)
        return Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_number_details(text: str) -> tuple[Fraction | None, str | None, str | None]:
    marked = _MARKED_NUMBER_RE.findall(text)
    candidates = marked or _NUMBER_RE.findall(text)
    parsed = [(raw, _fraction(raw)) for raw in candidates]
    parsed = [(raw, value) for raw, value in parsed if value is not None]
    if not parsed:
        return None, "number_missing", None
    if not marked and len({value for _, value in parsed}) > 1:
        return None, "number_ambiguous", None
    raw, value = parsed[-1]
    return value, None, raw


def parse_number(text: str) -> tuple[Fraction | None, str | None]:
    value, error, _raw = _parse_number_details(text)
    return value, error


def _mentions(text: str, candidates: tuple[str, ...]) -> list[str]:
    folded = _fold(text)
    found = []
    for candidate in candidates:
        alias = _fold(candidate)
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", folded):
            found.append(candidate)
    return found


def _verify_json(text: str, spec: AnswerSpec) -> Verification:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped,
                          flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return Verification(False, 0.0, failure_code="json_invalid",
                            details={"error": str(exc)})
    if not isinstance(value, dict):
        return Verification(False, 0.0, parsed_answer=value, failure_code="json_not_object")
    schema = spec.json_schema or {}
    if set(value) != set(schema):
        return Verification(False, 0.0, parsed_answer=value, failure_code="json_keys")
    for key, kind in schema.items():
        expected_type = {"integer": int, "string": str, "boolean": bool}.get(kind)
        if expected_type is None or type(value[key]) is not expected_type:
            return Verification(False, 0.0, parsed_answer=value, failure_code="json_type",
                                details={"key": key, "expected": kind})
    ok = value == spec.value
    return Verification(ok, float(ok), format_score=1.0, parsed_answer=value,
                        failure_code=None if ok else "json_value")


_ALLOWED_AST = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AnnAssign, ast.Expr, ast.If, ast.For, ast.While, ast.Compare, ast.BoolOp,
    ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.ListComp,
    ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Pass,
)
_SAFE_CALLS = {"abs", "all", "any", "bool", "enumerate", "float", "int", "len",
               "list", "max", "min", "range", "reversed", "round", "sorted", "str",
               "sum", "tuple", "zip"}


def _code_source(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text)
    return (match.group(1) if match else text).strip()


def _code_safe(source: str, function_name: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, "code_syntax"
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if functions != [function_name]:
        return False, "code_function"
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            return False, f"code_node_{type(node).__name__}"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return False, "code_dunder"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False, "code_call_target"
            if node.func.id not in _SAFE_CALLS and node.func.id != function_name:
                return False, "code_call_forbidden"
    return True, None


_CODE_RUNNER = r"""
import json, sys
payload = json.loads(sys.stdin.read())
safe = {k: getattr(__builtins__, k) for k in payload['safe']}
scope = {'__builtins__': safe}
exec(compile(payload['source'], '<rl-code>', 'exec'), scope, scope)
fn = scope[payload['function']]
out = []
for args, expected in payload['tests']:
    try:
        value = fn(*args)
        out.append({'ok': value == expected, 'value': value})
    except Exception as exc:
        out.append({'ok': False, 'error': type(exc).__name__})
print(json.dumps(out, ensure_ascii=False))
"""


def _verify_code(text: str, spec: AnswerSpec) -> Verification:
    source = _code_source(text)
    function_name = spec.function_name or ""
    safe, error = _code_safe(source, function_name)
    if not safe:
        return Verification(False, 0.0, failure_code=error)
    payload = {"source": source, "function": function_name, "tests": spec.tests,
               "safe": sorted(_SAFE_CALLS)}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _CODE_RUNNER], input=json.dumps(payload),
            text=True, capture_output=True, timeout=2.0, check=False,
        )
    except subprocess.TimeoutExpired:
        return Verification(False, 0.0, failure_code="code_timeout")
    if proc.returncode != 0:
        return Verification(False, 0.0, failure_code="code_runtime",
                            details={"stderr": proc.stderr[-300:]})
    try:
        results = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return Verification(False, 0.0, failure_code="code_protocol")
    passed = sum(bool(result.get("ok")) for result in results)
    fraction = passed / max(1, len(results))
    all_ok = passed == len(results)
    score = 0.7 * fraction + 0.3 * float(all_ok)
    return Verification(all_ok, score, format_score=1.0, parsed_answer=source,
                        failure_code=None if all_ok else "code_tests",
                        details={"passed": passed, "total": len(results)})


def verify(spec: AnswerSpec, generation: str) -> Verification:
    text = final_text(generation)
    if not text:
        return Verification(False, 0.0, failure_code="empty_final")
    if spec.kind in ("integer", "rational"):
        value, error, raw = _parse_number_details(text)
        if error:
            return Verification(False, 0.0, failure_code=error)
        expected = _fraction(str(spec.value))
        integer_format = raw is not None and not any(char in raw for char in ".,/")
        ok = value == expected and (spec.kind != "integer" or integer_format)
        return Verification(ok, float(ok), format_score=1.0, parsed_answer=str(value),
                            failure_code=None if ok else "number_wrong")
    if spec.kind in ("choice", "entity"):
        aliases = tuple(dict.fromkeys((str(spec.value), *spec.aliases)))
        universe = spec.choices or aliases
        found = _mentions(text, universe)
        expected_found = bool(_mentions(text, aliases))
        conflicts = [item for item in found if item not in aliases]
        ok = expected_found and not conflicts
        failure = None if ok else ("choice_conflict" if conflicts else "choice_missing")
        return Verification(ok, float(ok), format_score=1.0, parsed_answer=found,
                            failure_code=failure)
    if spec.kind == "abstain":
        ok = bool(_ABSTAIN_RE.search(_fold(text)))
        return Verification(ok, float(ok), format_score=1.0,
                            parsed_answer="abstain" if ok else None,
                            failure_code=None if ok else "abstain_missing")
    if spec.kind == "json":
        return _verify_json(text, spec)
    if spec.kind == "code":
        return _verify_code(text, spec)
    raise ValueError(f"type de réponse inconnu : {spec.kind}")
