"""Validate the module-based experiment archive layout without network access."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "p0-foundation-baselines",
    "p1-data-contract",
    "p2-shared-peft",
    "p3-static-backbone",
    "p4-expert-complementarity",
    "p5-contextual-routing",
    "secondary-smoothquant",
}
SESSION = re.compile(r"^\d{8}-[a-z0-9][a-z0-9_-]*$")


def module_sessions(root: Path) -> dict[str, set[str]]:
    found = {}
    actual_modules = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_modules != MODULES:
        raise ValueError(f"module mismatch under {root}: {sorted(actual_modules ^ MODULES)}")
    for module in sorted(MODULES):
        sessions = {
            path.name
            for path in (root / module).iterdir()
            if path.is_dir()
        }
        invalid = sorted(name for name in sessions if not SESSION.fullmatch(name))
        if invalid:
            raise ValueError(f"invalid session names under {module}: {invalid}")
        found[module] = sessions
    return found


def main() -> None:
    deprecated = [ROOT / "reports/sessions"]
    deprecated.extend(
        path
        for path in (ROOT / "results").iterdir()
        if path.is_dir() and path.name not in {"experiments", "shared", "indexes"}
    )
    existing = [path.relative_to(ROOT).as_posix() for path in deprecated if path.exists()]
    if existing:
        raise ValueError(f"deprecated experiment roots remain: {existing}")

    report_sessions = module_sessions(ROOT / "reports/experiments")
    result_sessions = module_sessions(ROOT / "results/experiments")
    for module, sessions in report_sessions.items():
        missing = sessions - result_sessions[module]
        if missing:
            raise ValueError(f"reports without matching results in {module}: {sorted(missing)}")

    backbone_path = ROOT / "configs/backbones/awq_w2a16_12l_mixed_spatial_v1.json"
    backbone = json.loads(backbone_path.read_text(encoding="utf-8"))
    language = backbone["language"]
    if language["w4_blocks"] != list(range(8, 16)) + list(range(20, 24)):
        raise ValueError("primary backbone W4 block set changed")
    if language["first_peft_target_blocks"] != [18, 19]:
        raise ValueError("primary PEFT target blocks changed")
    if backbone["vision"]["dino"] != {"weight_bits": 2, "group_size": 64}:
        raise ValueError("DINO backbone contract changed")
    if backbone["vision"]["siglip"] != {"weight_bits": 2, "group_size": 128}:
        raise ValueError("SigLIP backbone contract changed")

    for path in ROOT.rglob("*.json"):
        if ".git" not in path.parts:
            json.loads(path.read_text(encoding="utf-8-sig"))

    print(json.dumps({
        "modules": sorted(MODULES),
        "report_sessions": sum(map(len, report_sessions.values())),
        "result_sessions": sum(map(len, result_sessions.values())),
        "primary_backbone": backbone["backbone_id"],
        "status": "ok",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
