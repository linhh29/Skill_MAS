"""Minimal runtime for loading Skill-MAS specs for VitaBench."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_FRONTMATTER = ["name", "description", "tags", "inputs"]

# Skill-MAS 4-phase bank layout: ``skill_dir/phase_1/<pkg>/SKILL.md`` … ``phase_4/…``
MAS_PHASE_COUNT = 4


@dataclass
class SkillSpec:
    name: str
    description: str
    tags: list[str]
    inputs: list[str]
    body: str
    sections: Dict[str, str]
    validation_issues: list[str]
    path: Path


def _extract_sections(body: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    lines = body.splitlines()
    current: Optional[str] = None
    buf: list[str] = []
    for line in lines:
        m = re.match(r"^##\s+(.+)$", line.strip())
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            raw_title = m.group(1).strip()
            current = re.sub(r"[^a-z0-9]+", "_", raw_title.lower()).strip("_")
            buf = []
            continue
        if current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def _parse_frontmatter_raw(raw: str) -> Dict[str, Any]:
    """Parse simple YAML frontmatter with a lightweight fallback."""
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    data: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if re.match(r"^\s*-\s+", line) and current_list_key:
            item = re.sub(r"^\s*-\s+", "", line).strip()
            data.setdefault(current_list_key, [])
            if isinstance(data[current_list_key], list):
                data[current_list_key].append(item)
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = []
            current_list_key = key
        else:
            data[key] = value.strip("'\"")
    return data


def split_skill_document(text: str) -> tuple[dict[str, Any], str]:
    """Split SKILL.md ``text`` into frontmatter mapping and Markdown body."""
    frontmatter: Dict[str, Any] = {}
    body = text.strip() if isinstance(text, str) else ""
    raw = body
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            _, fm_raw, body_raw = parts
            frontmatter = _parse_frontmatter_raw(fm_raw)
            body = body_raw.strip()
    return frontmatter, body


def _candidate_frontmatter_key_usable(key: str, val: Any) -> bool:
    if val is None:
        return False
    if key in ("name", "description"):
        return isinstance(val, str) and bool(val.strip())
    if key in ("tags", "inputs"):
        seq = val if isinstance(val, list) else [val]
        return any(str(x).strip() for x in seq)
    return False


def _compose_skill_yaml_frontmatter(frontmatter: dict[str, Any]) -> str:
    """Serialize frontmatter mapping to a YAML ``--- ... ---`` block (no trailing body)."""
    try:
        import yaml  # type: ignore

        dumped = yaml.safe_dump(
            frontmatter,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=10_000,
        ).rstrip()
        return f"---\n{dumped}\n---"
    except Exception:
        nm = str(frontmatter.get("name") or "").strip() or "merged_skill"
        desc = str(frontmatter.get("description") or "").strip().replace('"', '\\"')
        tags_raw = frontmatter.get("tags") or []
        inputs_raw = frontmatter.get("inputs") or []
        tags = tags_raw if isinstance(tags_raw, list) else [tags_raw]
        inputs = inputs_raw if isinstance(inputs_raw, list) else [inputs_raw]
        tag_lines = "\n".join(f"  - {str(t).strip()}" for t in tags if str(t).strip()) or "  - meta-agent"
        in_lines = "\n".join(f"  - {str(i).strip()}" for i in inputs if str(i).strip()) or "  - user_query"
        extra_lines: list[str] = []
        for k in sorted(k for k in frontmatter.keys() if k not in REQUIRED_FRONTMATTER):
            v = frontmatter[k]
            if isinstance(v, (dict, list)):
                continue
            if v is None:
                continue
            extra_lines.append(f"{k}: {v}")
        extra = ("\n".join(extra_lines) + "\n") if extra_lines else ""
        return (
            "---\n"
            f"name: {nm}\n"
            f'description: "{desc}"\n'
            "tags:\n"
            f"{tag_lines}\n"
            "inputs:\n"
            f"{in_lines}\n"
            f"{extra}"
            "---"
        )


def merge_skill_generated_with_baseline_frontmatter(baseline_skill_text: str, candidate_skill_text: str) -> str:
    """
    When the optimizer model drops or truncates YAML, recover required keys from the baseline SKILL.md.

    Candidate body sections are preserved; merged frontmatter is baseline fields overridden by candidate
    when each required field is usable.
    """
    base_fm, _ = split_skill_document(baseline_skill_text)
    cand_fm, cand_body = split_skill_document(candidate_skill_text)
    merged: Dict[str, Any] = dict(base_fm)
    for key in REQUIRED_FRONTMATTER:
        if _candidate_frontmatter_key_usable(key, cand_fm.get(key)):
            merged[key] = cand_fm[key]
    return _compose_skill_yaml_frontmatter(merged) + "\n\n" + cand_body.strip() + ("\n" if cand_body.strip() else "")


def parse_skill_file(path: Path) -> SkillSpec:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = split_skill_document(text)

    name = str(frontmatter.get("name") or path.parent.name).strip()
    description = str(frontmatter.get("description") or "").strip()
    tags_raw = frontmatter.get("tags") or []
    inputs_raw = frontmatter.get("inputs") or []
    tags = [str(t).strip() for t in (tags_raw if isinstance(tags_raw, list) else [tags_raw]) if str(t).strip()]
    inputs = [str(i).strip() for i in (inputs_raw if isinstance(inputs_raw, list) else [inputs_raw]) if str(i).strip()]

    sections = _extract_sections(body)
    issues: list[str] = []
    for key in REQUIRED_FRONTMATTER:
        val = frontmatter.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            issues.append(f"missing_frontmatter:{key}")
    if not body.strip():
        issues.append("empty_body")
    if not sections:
        issues.append("no_h2_sections")

    return SkillSpec(
        name=name,
        description=description,
        tags=tags,
        inputs=inputs,
        body=body,
        sections=sections,
        validation_issues=issues,
        path=path,
    )


def discover_skills(
    skill_dir: str | Path, *, skip_invalid: bool = True
) -> tuple[list[SkillSpec], dict[str, list[str]]]:
    root = Path(skill_dir).resolve()
    if not root.exists():
        return [], {}
    files = sorted(root.glob("**/SKILL.md"))
    all_specs = [parse_skill_file(p) for p in files]
    issues = {s.name: s.validation_issues for s in all_specs if s.validation_issues}
    if not skip_invalid:
        return all_specs, issues
    return [s for s in all_specs if not s.validation_issues], issues


def is_phase_bank_layout(skill_dir: str | Path) -> bool:
    root = Path(skill_dir).resolve()
    # Prefer unified flat mode when root SKILL.md exists.
    if (root / "SKILL.md").is_file():
        return False
    return root.is_dir() and (root / "phase_1").is_dir()


def phase_bank_snapshot(
    skill_dir: str | Path,
    *,
    skip_invalid: bool = True,
    banks: Optional[dict[int, list[SkillSpec]]] = None,
    validation_issues: Optional[dict[str, list[str]]] = None,
) -> dict[str, Any]:
    """
    Structured description of the skill workspace for logging (phase banks vs legacy flat).

    If ``banks`` / ``validation_issues`` are already loaded, pass them to avoid a second scan.
    """
    root = Path(skill_dir).resolve()
    if banks is None:
        banks, val_iss = discover_phase_skill_banks(skill_dir, skip_invalid=skip_invalid)
    else:
        val_iss = validation_issues or {}
    return {
        "layout": "phase_banks" if is_phase_bank_layout(skill_dir) else "flat_replicated_to_phases",
        "skill_dir": str(root),
        "mas_phase_count": MAS_PHASE_COUNT,
        "phase_counts": {str(i): len(banks.get(i) or []) for i in range(1, MAS_PHASE_COUNT + 1)},
        "skills_by_phase": {
            str(i): [s.name for s in (banks.get(i) or [])] for i in range(1, MAS_PHASE_COUNT + 1)
        },
        "validation_issue_skill_names": sorted(val_iss.keys()),
    }


def discover_phase_skill_banks(
    skill_dir: str | Path, *, skip_invalid: bool = True
) -> tuple[dict[int, list[SkillSpec]], dict[str, list[str]]]:
    """
    Load per-phase skill banks.

    If ``skill_dir/phase_1`` exists, each ``phase_k`` directory holds one or more skill packages.
    Otherwise (legacy), the same flat skill list is replicated for all four phases.
    """
    root = Path(skill_dir).resolve()
    merged_issues: dict[str, list[str]] = {}
    banks: dict[int, list[SkillSpec]] = {i: [] for i in range(1, MAS_PHASE_COUNT + 1)}
    if not root.is_dir():
        return banks, merged_issues
    if is_phase_bank_layout(root):
        for i in range(1, MAS_PHASE_COUNT + 1):
            pdir = root / f"phase_{i}"
            if not pdir.is_dir():
                continue
            specs, iss = discover_skills(pdir, skip_invalid=skip_invalid)
            banks[i] = specs
            merged_issues.update(iss)
        return banks, merged_issues
    specs, iss = discover_skills(root, skip_invalid=skip_invalid)
    merged_issues.update(iss)
    for i in range(1, MAS_PHASE_COUNT + 1):
        banks[i] = list(specs)
    return banks, merged_issues


def build_injected_skill_text(selected_skills: list[SkillSpec]) -> str:
    if not selected_skills:
        return ""
    chunks: list[str] = []
    for idx, s in enumerate(selected_skills, start=1):
        chunks.append(
            "\n".join(
                [
                    f"### Skill {idx}: {s.name}",
                    f"- Description: {s.description}",
                    f"- Tags: {', '.join(s.tags) if s.tags else 'N/A'}",
                    f"- Inputs: {', '.join(s.inputs) if s.inputs else 'N/A'}",
                    "",
                    s.body.strip(),
                ]
            ).strip()
        )
    return "\n\n".join(chunks).strip()

