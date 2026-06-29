"""Extractor de atributos en capas (port del motor base TF-Final-independiente).

Reemplaza la extracción por listas regex cortas anteriores. Produce 9 capas
semánticas reales (TYPE, SUBTYPE, ACCESSORY, SHAPE, FEATURE, MATERIAL, COLOR,
CAPACITY, MOUTH_SIZE) más confianza por atributo, de forma determinista y
configurable por JSON. El hook opcional de Gemini (app/core/attribute_llm.py)
reutiliza los helpers de reglas/tokens de este módulo.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import pandas as pd

from app.core.text import normalize_text


DEFAULT_RULES_VERSION = "2026-06-11"
ALLOWED_RULE_METHODS = {"keyword_first_match", "keyword_any", "contains_any", "not_contains_any"}
CONFIGURABLE_ATTRIBUTES = {
    "product_type",
    "subtype",
    "accessory",
    "shape",
    "feature",
    "use_category",
}
AUTO_MERGE_LLM_ATTRIBUTES = {"subtype", "accessory", "shape", "feature"}
AUTO_MERGE_LLM_MATERIALS = False
AUTO_MERGE_LLM_COLORS = False

STOPWORDS = {
    "A", "AL", "C", "CADA", "CON", "DE", "DEL", "EL", "EN", "LA", "LAS", "LOS",
    "PACK", "PARA", "POR", "SIN", "UND", "UNID", "UNIDAD", "UNIDADES", "X", "Y",
}

COLORS = {
    "AMARILLO", "AMBAR", "AZUL", "BEIGE", "BLANCO", "CELESTE", "DORADO",
    "FOSFORESCENTE", "FUCSIA", "GRIS", "MARRON", "MORADO", "NARANJA", "NATURAL",
    "NEGRO", "PLATEADO", "ROJO", "ROSADO", "TRANSPARENTE", "VERDE",
}

MATERIAL_RULES = [
    ("POLICARBONATO", "PC", "POLICARBONATO"),
    ("ACRILICO", "ACRILICO", "ACRILICO"),
    ("ALUMINIO", "ALUMINIO", "METAL"),
    ("CARTON", "CARTON", "PAPEL_CARTON"),
    ("LDPE", "LDPE", "POLIETILENO"),
    ("PEAD", "PEAD", "POLIETILENO"),
    ("PETG", "PETG", "POLIETILENO TEREFTALATO"),
    ("PET", "PET", "POLIETILENO TEREFTALATO"),
    ("PAPEL", "PAPEL", "PAPEL_CARTON"),
    ("METAL", "METAL", "METAL"),
    ("NYLON", "NYLON", "POLIAMIDA"),
    ("PLASTICO", "PLASTICO", "PLASTICO"),
    ("PP", "PP", "POLIPROPILENO"),
    ("PVC", "PVC", "PVC"),
    ("VIDRIO", "VIDRIO", "VIDRIO"),
]

PRODUCT_TYPE_RULES = [
    (("CREMERO", "VASELINERO"), "cremero", 0.94),
    (("PASTILLERO",), "accesorio", 0.90),
    (("PISETA", "GRADILLA", "TALQUERA", "PROBETA", "PROPIPETA"), "accesorio", 0.88),
    (("POTE", "CREMERO"), "pote", 0.92),
    (("TUBO", "TUBOS"), "tubo", 0.90),
    (("FRASCO", "BOTELLA", "ENVASE"), "frasco", 0.93),
    (("GOTERO",), "gotero", 0.95),
    (("BOMBA", "ATOMIZADORA", "ATOMIZADOR", "SPRAY", "GATILLO", "PULVERIZADOR", "TRIGGER"), "atomizador", 0.92),
    (("TAPA", "TAPON", "VALVULA", "DISPENSADOR", "ATOMIZADOR"), "tapa", 0.90),
    (("BOLSA", "BOLSAS", "ZIPLOC"), "bolsa", 0.91),
    (("CAJA", "CAJAS"), "caja", 0.90),
    (("KIT", "SET"), "kit", 0.86),
    (("ETIQUETA", "STICKER"), "etiqueta", 0.90),
    (("MAQUINA", "EQUIPO", "LAPTOP", "IMPRESORA"), "equipo", 0.84),
    (("REPUESTO", "ACCESORIO"), "accesorio", 0.84),
]

SUBTYPE_RULES = [
    (("AIRLESS",), "airless", 0.96),
    (("ATOMIZADOR", "SPRAY"), "atomizador", 0.95),
    (("BULLET",), "bullet", 0.97),
    (("CAMPANA",), "campana", 0.94),
    (("CHUPON",), "chupon", 0.94),
    (("CREMERO",), "cremero", 0.95),
    (("DISPENSADOR", "DISPENSADORA"), "dispensador", 0.95),
    (("ESPUMERO",), "espumero", 0.97),
    (("GATILLO",), "gatillo", 0.94),
    (("GOTERO",), "gotero", 0.95),
    (("LAINA",), "laina", 0.92),
    (("PASTILLERO",), "pastillero", 0.93),
    (("RIMEL",), "rimel", 0.95),
    (("ROSCA",), "rosca", 0.90),
    (("ZIPLOC",), "ziploc", 0.95),
]

USE_RULES = [
    (("RIMEL", "LIPSTICK", "CREMERO", "COSMET", "BULLET", "AIRLESS"), "cosmetica", 0.89),
    (("ODONTO", "MEDIC", "CLINIC", "ORTOPEDIA", "PH", "CLORO"), "medico_laboratorio", 0.84),
    (("PINTURA", "BROCHA", "TORNILLO", "TUERCA", "MANGUERA"), "ferreteria", 0.84),
    (("BOLSA", "CAJA", "ENVASE", "FRASCO", "POTE"), "empaque_envase", 0.82),
]

DEFAULT_ATTRIBUTE_RULES = {
    "rules_version": DEFAULT_RULES_VERSION,
    "domain": "envases_vidrio_plastico",
    "llm_status": "not_used",
    "allowed_methods": sorted(ALLOWED_RULE_METHODS),
    "attribute_rules": {
        "product_type": {
            "method": "keyword_first_match",
            "rules": [
                {"keywords": list(keywords), "value": value, "confidence": confidence}
                for keywords, value, confidence in PRODUCT_TYPE_RULES
            ],
        },
        "subtype": {
            "method": "keyword_first_match",
            "rules": [
                {"keywords": list(keywords), "value": value, "confidence": confidence}
                for keywords, value, confidence in SUBTYPE_RULES
            ]
            + [
                {"keywords": ["ESMALTE"], "value": "esmalte", "confidence": 0.92},
                {"keywords": ["BALSAMO", "LIPSTICK"], "value": "lipstick", "confidence": 0.90},
                {"keywords": ["VIAL"], "value": "vial", "confidence": 0.88},
                {"keywords": ["DOSIFICADOR"], "value": "dosificador", "confidence": 0.86},
            ],
        },
        "accessory": {
            "method": "keyword_any",
            "rules": [
                {"keywords": ["BROCHA"], "value": "brocha", "confidence": 0.92},
                {"keywords": ["CHUPON"], "value": "chupon", "confidence": 0.92},
                {"keywords": ["LAINA"], "value": "laina", "confidence": 0.90},
                {"keywords": ["TAPA", "TAPON"], "value": "tapa", "confidence": 0.88},
                {"keywords": ["REJILLA"], "value": "rejilla", "confidence": 0.88},
                {"keywords": ["GATILLO", "TRIGGER"], "value": "gatillo", "confidence": 0.90},
            ],
        },
        "shape": {
            "method": "keyword_first_match",
            "rules": [
                {"keywords": ["OVALADO", "OVALADA"], "value": "ovalado", "confidence": 0.90},
                {"keywords": ["CONICO", "CONICA"], "value": "conico", "confidence": 0.88},
                {"keywords": ["TUBULAR"], "value": "tubular", "confidence": 0.88},
                {"keywords": ["CILINDRICO", "CILINDRICA"], "value": "cilindrico", "confidence": 0.86},
            ],
        },
        "feature": {
            "method": "keyword_first_match",
            "rules": [
                {"keywords": ["NO ESTERIL"], "value": "no_esteril", "confidence": 0.88},
                {"keywords": ["ESTERIL", "ESTERILES"], "value": "esteril", "confidence": 0.90},
                {"keywords": ["GRADUADO", "GRADUADA"], "value": "graduado", "confidence": 0.90},
                {"keywords": ["DESCARTABLE"], "value": "descartable", "confidence": 0.88},
                {"keywords": ["IMPORTADO", "IMPORTADA"], "value": "importado", "confidence": 0.80},
            ],
        },
        "use_category": {
            "method": "keyword_first_match",
            "rules": [
                {"keywords": list(keywords), "value": value, "confidence": confidence}
                for keywords, value, confidence in USE_RULES
            ],
        },
    },
    "material_rules": [
        {"keyword": keyword, "material": material, "family": family, "confidence": 0.94}
        for keyword, material, family in MATERIAL_RULES
    ],
    "color_keywords": sorted(COLORS),
}


# ── Tokenización y reglas ───────────────────────────────────────────────────


def tokenize_description(value: object) -> list[str]:
    text = normalize_text(value)
    return [token for token in text.split() if token and token not in STOPWORDS]


def normalize_rule_keywords(rule: dict[str, Any]) -> list[str]:
    raw_keywords = rule.get("keywords", rule.get("match", []))
    if isinstance(raw_keywords, str):
        raw_keywords = [raw_keywords]
    return [normalize_text(keyword) for keyword in raw_keywords if normalize_text(keyword)]


def phrase_or_token_in_text(keyword: str, text: str, tokens: set[str]) -> bool:
    if " " in keyword:
        return keyword in text
    return keyword in tokens


def apply_configured_attribute_rule(text: str, rule_def: dict[str, Any]) -> tuple[str | None, float, str]:
    method = rule_def.get("method", "keyword_first_match")
    if method not in ALLOWED_RULE_METHODS:
        return None, 0.0, ""

    tokens = set(text.split())
    matched_values: list[tuple[str, float, str]] = []
    for rule in rule_def.get("rules", []):
        keywords = normalize_rule_keywords(rule)
        if not keywords:
            continue
        confidence = float(rule.get("confidence", 0.85))
        value = str(rule.get("value", "")).strip()
        if not value:
            continue

        has_match = any(phrase_or_token_in_text(keyword, text, tokens) for keyword in keywords)
        if method == "not_contains_any":
            has_match = not has_match
        if not has_match:
            continue
        if method == "keyword_first_match":
            return value, confidence, ",".join(keywords)
        matched_values.append((value, confidence, ",".join(keywords)))

    if matched_values:
        values: list[str] = []
        confidences: list[float] = []
        evidence: list[str] = []
        for value, confidence, keyword in matched_values:
            if value not in values:
                values.append(value)
                confidences.append(confidence)
                evidence.append(keyword)
        return "|".join(values), max(confidences), "|".join(evidence)
    return None, 0.0, ""


def validate_attribute_rules(rules: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(rules, dict):
        raise ValueError("attribute_rules debe ser un objeto JSON.")

    merged = json.loads(json.dumps(DEFAULT_ATTRIBUTE_RULES, ensure_ascii=False))
    for key in ("rules_version", "domain", "llm_status"):
        if key in rules:
            merged[key] = rules[key]

    incoming_attribute_rules = rules.get("attribute_rules", {})
    if not isinstance(incoming_attribute_rules, dict):
        raise ValueError("attribute_rules.attribute_rules debe ser un objeto.")

    for attr, rule_def in incoming_attribute_rules.items():
        if attr not in CONFIGURABLE_ATTRIBUTES:
            continue
        if not isinstance(rule_def, dict):
            continue
        method = rule_def.get("method", "keyword_first_match")
        if method not in ALLOWED_RULE_METHODS:
            continue
        safe_rules = []
        for rule in rule_def.get("rules", []):
            if not isinstance(rule, dict):
                continue
            keywords = normalize_rule_keywords(rule)
            value = str(rule.get("value", "")).strip()
            if not keywords or not value:
                continue
            confidence = min(max(float(rule.get("confidence", 0.85)), 0.0), 1.0)
            safe_rules.append({"keywords": keywords, "value": value, "confidence": confidence})
        if safe_rules:
            merged["attribute_rules"][attr] = {"method": method, "rules": safe_rules}

    if isinstance(rules.get("material_rules"), list):
        safe_materials = []
        for rule in rules["material_rules"]:
            keyword = normalize_text(rule.get("keyword", ""))
            material = str(rule.get("material", "")).strip().upper()
            family = str(rule.get("family", material)).strip().upper()
            if keyword and material:
                safe_materials.append(
                    {
                        "keyword": keyword,
                        "material": material,
                        "family": family,
                        "confidence": min(max(float(rule.get("confidence", 0.94)), 0.0), 1.0),
                    }
                )
        if safe_materials:
            merged["material_rules"] = safe_materials

    if isinstance(rules.get("color_keywords"), list):
        colors = sorted(
            {
                color
                for color in (normalize_text(raw_color) for raw_color in rules["color_keywords"])
                if re.fullmatch(r"[A-ZÑ]{3,24}", color)
            }
        )
        if colors:
            merged["color_keywords"] = colors
    return merged


def sanitize_attribute_rule_additions(rules: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(rules, dict):
        raise ValueError("Las reglas propuestas por LLM deben ser un objeto JSON.")

    sanitized = {
        "rules_version": rules.get("rules_version", DEFAULT_RULES_VERSION),
        "domain": rules.get("domain", "unknown"),
        "llm_status": rules.get("llm_status", "proposed_additions"),
        "allowed_methods": sorted(ALLOWED_RULE_METHODS),
        "attribute_rules": {},
        "material_rules": [],
        "color_keywords": [],
    }

    incoming_attribute_rules = rules.get("attribute_rules", {})
    if isinstance(incoming_attribute_rules, dict):
        for attr, rule_def in incoming_attribute_rules.items():
            if attr not in AUTO_MERGE_LLM_ATTRIBUTES:
                continue
            if not isinstance(rule_def, dict):
                continue
            method = rule_def.get("method", "keyword_first_match")
            if method not in ALLOWED_RULE_METHODS:
                continue
            safe_rules = []
            for rule in rule_def.get("rules", []):
                if not isinstance(rule, dict):
                    continue
                keywords = normalize_rule_keywords(rule)
                value = normalize_text(rule.get("value", "")).lower().replace(" ", "_")
                if not keywords or not value:
                    continue
                confidence = min(max(float(rule.get("confidence", 0.85)), 0.0), 1.0)
                safe_rules.append({"keywords": keywords, "value": value, "confidence": confidence})
            if safe_rules:
                sanitized["attribute_rules"][attr] = {"method": method, "rules": safe_rules}

    if isinstance(rules.get("material_rules"), list):
        for rule in rules["material_rules"]:
            keyword = normalize_text(rule.get("keyword", ""))
            material = str(rule.get("material", "")).strip().upper()
            family = str(rule.get("family", material)).strip().upper()
            if keyword and material:
                sanitized["material_rules"].append(
                    {
                        "keyword": keyword,
                        "material": material,
                        "family": family,
                        "confidence": min(max(float(rule.get("confidence", 0.94)), 0.0), 1.0),
                    }
                )

    if isinstance(rules.get("color_keywords"), list):
        sanitized["color_keywords"] = sorted(
            {
                color
                for color in (normalize_text(raw_color) for raw_color in rules["color_keywords"])
                if re.fullmatch(r"[A-ZÑ]{3,24}", color)
            }
        )
    return sanitized


def merge_attribute_rules(base_rules: dict[str, Any], additional_rules: dict[str, Any]) -> dict[str, Any]:
    merged = validate_attribute_rules(base_rules)
    additions = sanitize_attribute_rule_additions(additional_rules)

    for attr, add_def in additions.get("attribute_rules", {}).items():
        if attr not in AUTO_MERGE_LLM_ATTRIBUTES:
            continue
        base_def = merged["attribute_rules"].setdefault(
            attr, {"method": add_def.get("method", "keyword_first_match"), "rules": []}
        )
        existing = {
            (tuple(rule.get("keywords", [])), rule.get("value"))
            for rule in base_def.get("rules", [])
        }
        for rule in add_def.get("rules", []):
            key = (tuple(rule.get("keywords", [])), rule.get("value"))
            if key not in existing:
                base_def["rules"].append(rule)
                existing.add(key)

    if AUTO_MERGE_LLM_MATERIALS:
        existing_materials = {
            (rule.get("keyword"), rule.get("material")) for rule in merged.get("material_rules", [])
        }
        for rule in additions.get("material_rules", []):
            key = (rule.get("keyword"), rule.get("material"))
            if key not in existing_materials:
                merged["material_rules"].append(rule)
                existing_materials.add(key)

    if AUTO_MERGE_LLM_COLORS:
        merged["color_keywords"] = sorted(
            set(merged.get("color_keywords", [])) | set(additions.get("color_keywords", []))
        )
    merged["llm_status"] = additional_rules.get("llm_status", "merged_additions")
    return merged


def load_attribute_rules(rules: dict[str, Any] | None = None) -> dict[str, Any]:
    if rules is None:
        return validate_attribute_rules(DEFAULT_ATTRIBUTE_RULES)
    return validate_attribute_rules(rules)


# ── Extractores numéricos / textuales ───────────────────────────────────────


def extract_capacity(text: str) -> tuple[float | None, str | None, float, str]:
    pattern = r"\b(\d+(?:[\.,]\d+)?)\s*(ML|CC|L|LT|LTR|GR|G|KG|MG|W|V)\b"
    match = re.search(pattern, text)
    if not match:
        return None, None, 0.0, ""
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2)
    if unit in {"LT", "LTR"}:
        unit = "L"
    if unit == "G":
        unit = "GR"
    return value, unit, 0.95, match.group(0)


def extract_mouth_size(text: str) -> tuple[float | None, float, str]:
    patterns = [
        r"\b[BTN]\s*([0-9]{2,3})\s*(?:MM)?\b",
        r"\b([0-9]{2,3})\s*/\s*410\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), 0.92, match.group(0)
    return None, 0.0, ""


def extract_size_text(text: str) -> tuple[str | None, float, str]:
    pattern = r"\b\d+(?:[\.,]\d+)?\s*(?:CM|MM|M)?\s*[*X]\s*\d+(?:[\.,]\d+)?\s*(?:CM|MM|M)?\b"
    match = re.search(pattern, text)
    if not match:
        return None, 0.0, ""
    return match.group(0), 0.93, match.group(0)


def extract_color(text: str, rules: dict[str, Any] | None = None) -> tuple[str | None, float, str]:
    tokens = text.split()
    colors = rules.get("color_keywords", sorted(COLORS)) if rules else sorted(COLORS)
    for color in colors:
        if color in tokens:
            return color, 0.94, color
    return None, 0.0, ""


def extract_material(text: str, rules: dict[str, Any] | None = None) -> tuple[str | None, str | None, float, str]:
    tokens = set(text.split())
    material_rules = (
        rules.get("material_rules", DEFAULT_ATTRIBUTE_RULES["material_rules"])
        if rules
        else DEFAULT_ATTRIBUTE_RULES["material_rules"]
    )
    for rule in material_rules:
        keyword = normalize_text(rule.get("keyword", ""))
        material = rule.get("material")
        family = rule.get("family", material)
        if keyword in tokens:
            return material, family, float(rule.get("confidence", 0.94)), keyword
    return None, None, 0.0, ""


def extract_attributes_for_product(product: pd.Series, rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or load_attribute_rules()
    product_name = product.get("product_name", "")
    text = normalize_text(product_name)
    tokens = tokenize_description(product_name)

    capacity_value, capacity_unit, capacity_conf, capacity_evidence = extract_capacity(text)
    mouth_size, mouth_conf, mouth_evidence = extract_mouth_size(text)
    size_text, size_conf, size_evidence = extract_size_text(text)
    color, color_conf, color_evidence = extract_color(text, rules)
    material, material_family, material_conf, material_evidence = extract_material(text, rules)
    attr_rules = rules.get("attribute_rules", {})
    product_type, type_conf, type_evidence = apply_configured_attribute_rule(text, attr_rules.get("product_type", {}))
    subtype, subtype_conf, subtype_evidence = apply_configured_attribute_rule(text, attr_rules.get("subtype", {}))
    accessory, accessory_conf, accessory_evidence = apply_configured_attribute_rule(text, attr_rules.get("accessory", {}))
    shape, shape_conf, shape_evidence = apply_configured_attribute_rule(text, attr_rules.get("shape", {}))
    feature, feature_conf, feature_evidence = apply_configured_attribute_rule(text, attr_rules.get("feature", {}))
    use_category, use_conf, use_evidence = apply_configured_attribute_rule(text, attr_rules.get("use_category", {}))

    if not use_category:
        use_category = "general"
        use_conf = 0.60
        use_evidence = "default"

    confidence_values = [
        value
        for value in (
            capacity_conf, mouth_conf, size_conf, color_conf, material_conf,
            type_conf, subtype_conf, accessory_conf, shape_conf, feature_conf, use_conf,
        )
        if value > 0
    ]
    avg_confidence = sum(confidence_values) / max(len(confidence_values), 1)
    extraction_method = "deterministic_rules" if confidence_values else "unclassified"
    attribute_confidence = {
        attr: round(conf, 4)
        for attr, conf in (
            ("product_type", type_conf),
            ("subtype", subtype_conf),
            ("accessory", accessory_conf),
            ("shape", shape_conf),
            ("feature", feature_conf),
            ("material", material_conf),
            ("color", color_conf),
            ("capacity", capacity_conf),
            ("mouth_size", mouth_conf),
            ("size_text", size_conf),
            ("use_category", use_conf),
        )
        if conf > 0
    }
    evidence = {
        "product_type": type_evidence,
        "subtype": subtype_evidence,
        "accessory": accessory_evidence,
        "shape": shape_evidence,
        "feature": feature_evidence,
        "material": material_evidence,
        "color": color_evidence,
        "capacity": capacity_evidence,
        "mouth_size": mouth_evidence,
        "size_text": size_evidence,
        "use_category": use_evidence,
    }

    return {
        "product_id": product.get("product_id"),
        "product_name": product_name,
        "product_type": product_type,
        "subtype": subtype,
        "accessory": accessory,
        "shape": shape,
        "feature": feature,
        "material": material,
        "color": color,
        "capacity_value": capacity_value,
        "capacity_unit": capacity_unit,
        "mouth_size_mm": mouth_size,
        "size_text": size_text,
        "use_category": use_category,
        "material_family": material_family,
        "keywords": json.dumps(tokens[:8], ensure_ascii=False),
        "confidence": round(avg_confidence, 4),
        "attribute_confidence": json.dumps(attribute_confidence, ensure_ascii=False),
        "extraction_method": extraction_method,
        "rules_version": rules.get("rules_version", DEFAULT_RULES_VERSION),
        "attribute_evidence": json.dumps(evidence, ensure_ascii=False),
    }


# ── Perfil de tokens (insumo del refinamiento LLM opcional) ──────────────────


def build_token_profile(products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    descriptions = products.get("product_name", pd.Series(dtype=str)).fillna("")
    product_count = len(descriptions)
    token_counter: Counter[str] = Counter()
    product_presence: Counter[str] = Counter()

    for description in descriptions:
        tokens = tokenize_description(description)
        token_counter.update(tokens)
        product_presence.update(set(tokens))

    for token, count in token_counter.most_common(100):
        rows.append(
            {
                "token": token,
                "token_count": int(count),
                "product_count": int(product_presence[token]),
                "product_coverage": round(product_presence[token] / max(product_count, 1), 4),
            }
        )
    return pd.DataFrame(rows)


def collect_rule_keywords(rules: dict[str, Any]) -> set[str]:
    covered: set[str] = set()
    for rule_def in rules.get("attribute_rules", {}).values():
        for rule in rule_def.get("rules", []):
            covered.update(normalize_rule_keywords(rule))
    for rule in rules.get("material_rules", []):
        keyword = normalize_text(rule.get("keyword", ""))
        if keyword:
            covered.add(keyword)
    covered.update(normalize_text(color) for color in rules.get("color_keywords", []))
    return covered


def build_uncovered_token_candidates(
    token_profile: pd.DataFrame,
    current_rules: dict[str, Any],
    limit: int = 50,
) -> list[dict[str, Any]]:
    covered = collect_rule_keywords(current_rules)
    ignored_patterns = [
        r"^\d+$",
        r"^\d+(ML|CC|L|LT|GR|G|KG|MG|MM|CM|UND|UNID)$",
        r"^[BTN]?\d{2,3}$",
    ]
    candidates = []
    for _, row in token_profile.iterrows():
        token = normalize_text(row["token"])
        if not token or token in covered or token in STOPWORDS:
            continue
        if any(re.fullmatch(pattern, token) for pattern in ignored_patterns):
            continue
        if len(token) <= 2:
            continue
        candidates.append(
            {
                "token": token,
                "token_count": int(row.get("token_count", 0)),
                "product_count": int(row.get("product_count", 0)),
                "product_coverage": float(row.get("product_coverage", 0)),
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def build_llm_additive_rules_prompt(
    descriptions: list[str],
    token_profile: pd.DataFrame,
    current_rules: dict[str, Any] | None = None,
    max_tokens: int = 45,
    max_descriptions: int = 25,
) -> str:
    rules = current_rules or validate_attribute_rules(DEFAULT_ATTRIBUTE_RULES)
    candidates = build_uncovered_token_candidates(token_profile, rules, limit=max_tokens)
    sample = descriptions[:max_descriptions]
    schema = {
        "rules_version": DEFAULT_RULES_VERSION,
        "domain": "short_domain_name",
        "llm_status": "proposed_additions",
        "attribute_rules": {
            "product_type": {"method": "keyword_first_match", "rules": []},
            "subtype": {"method": "keyword_first_match", "rules": []},
            "accessory": {"method": "keyword_any", "rules": []},
            "shape": {"method": "keyword_first_match", "rules": []},
            "feature": {"method": "keyword_first_match", "rules": []},
            "use_category": {"method": "keyword_first_match", "rules": []},
        },
        "material_rules": [],
        "color_keywords": [],
    }
    return (
        "Eres un asistente de normalizacion de catalogos empresariales.\n"
        "Debes proponer SOLO reglas NUEVAS que no esten ya cubiertas.\n"
        "No reconstruyas las reglas existentes. Si no hay reglas utiles, devuelve listas vacias.\n"
        "No escribas Python. No inventes productos. No agregues atributos fuera del esquema.\n"
        "Prioriza subtype, accessory, shape y feature, porque son las reglas que se aceptan automaticamente.\n"
        "Puedes proponer product_type, use_category, material_rules o color_keywords solo si son muy evidentes, "
        "pero esas reglas requeriran revision y no se autoaceptan.\n"
        "Usa keywords en MAYUSCULAS sin tildes. Usa values en espanol, minusculas y snake_case; no traduzcas al ingles.\n"
        "Devuelve SOLO JSON valido.\n"
        "Reglas estrictas de formato: comillas dobles, sin comentarios, sin trailing commas, "
        "sin markdown y sin texto fuera del JSON.\n\n"
        "Atributos permitidos:\n"
        "- product_type: objeto principal del producto.\n"
        "- subtype: variante comercial o familia especifica.\n"
        "- accessory: piezas incluidas o asociadas como tapa, brocha, chupon.\n"
        "- shape: forma fisica como ovalado, conico, tubular.\n"
        "- feature: cualidad como esteril, graduado, importado.\n"
        "- use_category: uso general del producto.\n"
        "- material_rules: materiales explicitos.\n"
        "- color_keywords: colores explicitos.\n\n"
        "Formato obligatorio de cada regla:\n"
        '{"keywords":["TOKEN"],"value":"valor_normalizado","confidence":0.85}\n\n'
        f"ESQUEMA DE SALIDA:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"TOKENS CANDIDATOS NO CUBIERTOS:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        f"MUESTRA DE DESCRIPCIONES:\n{json.dumps(sample, ensure_ascii=False, indent=2)}\n"
    )


# ── Fachada usada por el pipeline ───────────────────────────────────────────

# Columnas que cuentan como "atributo presente" para la cobertura.
COVERAGE_COLUMNS = [
    "product_type",
    "subtype",
    "accessory",
    "shape",
    "feature",
    "material",
    "color",
    "capacity_value",
    "mouth_size_mm",
]


def _filled(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip().replace({"None": ""}) != "")


def extract_product_attributes(
    products: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Extrae atributos en capas para todo el catálogo.

    Devuelve (DataFrame, report). El report mantiene las claves `coverage`
    (float) y `rules` (dict) que el pipeline usa para decidir si construye el
    grafo y para exportar `attribute_rules.json`.
    """
    active_rules = load_attribute_rules(rules)
    if products.empty:
        empty = pd.DataFrame()
        return empty, {
            "products": 0,
            "products_with_attributes": 0,
            "coverage": 0.0,
            "avg_confidence": 0.0,
            "rules": active_rules,
        }

    attributes = pd.DataFrame(
        [extract_attributes_for_product(row, active_rules) for _, row in products.iterrows()]
    )

    filled_any = pd.Series(False, index=attributes.index)
    coverage_by_attribute: dict[str, float] = {}
    total = len(attributes)
    for column in COVERAGE_COLUMNS:
        mask = _filled(attributes[column])
        coverage_by_attribute[column] = round(float(mask.sum()) / max(total, 1), 4)
        filled_any = filled_any | mask
    attributes["attribute_count"] = sum(_filled(attributes[col]).astype(int) for col in COVERAGE_COLUMNS)

    coverage = float(filled_any.mean()) if total else 0.0
    report = {
        "products": int(total),
        "products_with_attributes": int(filled_any.sum()),
        "coverage": round(coverage, 4),
        "avg_confidence": round(float(attributes["confidence"].mean()), 4) if total else 0.0,
        "coverage_by_attribute": coverage_by_attribute,
        "rules": active_rules,
    }
    return attributes, report
