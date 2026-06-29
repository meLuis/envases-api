"""Refinamiento opcional de reglas de atributos con Gemini (port adaptado).

Estrategia ADITIVA y conservadora (heredada del proyecto base):
- El pipeline regex/diccionario corre siempre. Gemini solo PROPONE reglas nuevas
  para tokens de alta frecuencia que aún no están cubiertos.
- Solo se auto-fusionan capas semánticas (subtype/accessory/shape/feature).
- El resultado se ACEPTA únicamente si no hay regresión: la cobertura de esas
  capas debe mejorar y ni la cobertura global ni la confianza media deben caer.

Todo está protegido: sin GEMINI_API_KEY / sin google-genai / ante cualquier
error, se devuelven las reglas base sin tocar (no bloquea la construcción del
dataset).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd

from app.core.attributes import (
    build_llm_additive_rules_prompt,
    build_token_profile,
    extract_product_attributes,
    load_attribute_rules,
    merge_attribute_rules,
    sanitize_attribute_rule_additions,
)

AUTO_LAYERS = ["subtype", "accessory", "shape", "feature"]
DEFAULT_SAMPLE_SIZE = 600
DEFAULT_TOKEN_LIMIT = 80


def gemini_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Falta GEMINI_API_KEY o GOOGLE_API_KEY.")
    return key


def call_gemini(prompt: str) -> str:
    from google import genai  # import perezoso; dependencia opcional

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=_api_key())
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or "{}"


def extract_json_object(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _sample_descriptions(products: pd.DataFrame, attributes: pd.DataFrame, sample_size: int) -> list[str]:
    products = products.copy()
    products["product_name"] = products.get("product_name", "").fillna("").astype(str)
    products["product_id"] = products["product_id"].astype(str)
    attrs = attributes.copy()
    attrs["product_id"] = attrs["product_id"].astype(str)
    scoped = products.merge(attrs, on="product_id", how="left", suffixes=("", "_attr"))
    for col in AUTO_LAYERS:
        if col not in scoped.columns:
            scoped[col] = pd.NA
    scoped["missing_signal"] = scoped[AUTO_LAYERS].isna().sum(axis=1)
    scoped["confidence"] = pd.to_numeric(scoped.get("confidence", 0), errors="coerce").fillna(0)

    low_structure = scoped.sort_values(["missing_signal", "confidence"], ascending=[False, True]).head(sample_size // 2)
    long_names = scoped.assign(name_len=scoped["product_name"].str.len()).sort_values(
        "name_len", ascending=False
    ).head(sample_size // 4)
    random_names = scoped.sample(min(sample_size, len(scoped)), random_state=43)
    combined = pd.concat([low_structure, long_names, random_names], ignore_index=True)
    combined = combined.drop_duplicates(subset=["product_name"]).head(sample_size)
    return combined["product_name"].tolist()


def _auto_layer_coverage(report: dict) -> float:
    cov = report.get("coverage_by_attribute", {})
    return sum(float(cov.get(layer, 0.0)) for layer in AUTO_LAYERS)


def refine_rules_with_gemini(
    products: pd.DataFrame,
    base_attributes: pd.DataFrame,
    base_report: dict,
) -> tuple[pd.DataFrame, dict, dict]:
    """Intenta mejorar las reglas con Gemini, de forma conservadora.

    Devuelve (attributes, report, info). Si Gemini no está disponible o no hay
    mejora sin regresión, devuelve los `base_*` sin cambios. `info` describe la
    decisión para registrarla en el dataset.
    """
    if not gemini_available():
        return base_attributes, base_report, {"status": "disabled", "reason": "sin GEMINI_API_KEY"}

    try:
        base_rules = base_report.get("rules") or load_attribute_rules()
        token_profile = build_token_profile(products)
        descriptions = _sample_descriptions(products, base_attributes, DEFAULT_SAMPLE_SIZE)
        prompt = build_llm_additive_rules_prompt(
            descriptions,
            token_profile,
            base_rules,
            max_tokens=int(os.environ.get("LLM_TOKEN_LIMIT", str(DEFAULT_TOKEN_LIMIT))),
            max_descriptions=DEFAULT_SAMPLE_SIZE,
        )
        raw = call_gemini(prompt)
        additions = sanitize_attribute_rule_additions(extract_json_object(raw))
        additions["llm_status"] = "additions_proposed_by_gemini"
        additions["source_model"] = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        proposed_count = sum(len(d.get("rules", [])) for d in additions.get("attribute_rules", {}).values())
        if proposed_count == 0:
            return base_attributes, base_report, {
                "status": "rejected",
                "reason": "Gemini no propuso reglas nuevas",
                "source_model": additions["source_model"],
            }

        merged_rules = merge_attribute_rules(base_rules, additions)
        cand_attributes, cand_report = extract_product_attributes(products, merged_rules)

        # Aceptación conservadora: mejorar capas semánticas sin regresión global.
        improves = _auto_layer_coverage(cand_report) > _auto_layer_coverage(base_report)
        no_global_regression = cand_report["coverage"] >= base_report["coverage"] - 1e-9
        no_confidence_regression = cand_report["avg_confidence"] >= base_report["avg_confidence"] * 0.99
        accepted = improves and no_global_regression and no_confidence_regression

        info = {
            "status": "accepted" if accepted else "rejected",
            "source_model": additions["source_model"],
            "rules_proposed": proposed_count,
            "base_auto_layer_coverage": round(_auto_layer_coverage(base_report), 4),
            "candidate_auto_layer_coverage": round(_auto_layer_coverage(cand_report), 4),
            "base_coverage": base_report["coverage"],
            "candidate_coverage": cand_report["coverage"],
        }
        if accepted:
            cand_report["rules"] = merged_rules
            return cand_attributes, cand_report, info
        return base_attributes, base_report, info
    except Exception as exc:  # noqa: BLE001 — refinamiento es accesorio
        return base_attributes, base_report, {"status": "error", "reason": str(exc)}
