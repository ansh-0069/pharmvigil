"""
Seed sample CAPA cases across all workflow stages for demo purposes.
Run once: python scripts/seed_capa.py
"""
import os
os.environ.setdefault("USE_SQLITE", "1")

from src.services.capa_service import create_capa_case, transition_to_investigation, transition_to_corrective_action, transition_to_verification, close_capa_case

CASES = [
    {"product_id": "PROD-001", "title": "Contamination in Batch #42",          "description": "Microbial contamination detected during QC inspection of Batch #42.", "priority": "CRITICAL", "assigned_to": "Dr. Patel",    "advance_to": None},
    {"product_id": "PROD-002", "title": "Mislabelling on packaging line",      "description": "Wrong dosage printed on 200 units of Product-002.",                  "priority": "HIGH",     "assigned_to": "Sarah Kim",   "advance_to": "INVESTIGATION"},
    {"product_id": "PROD-001", "title": "Temperature excursion in cold chain", "description": "Storage temperature breached +8°C threshold for 3 hours.",           "priority": "HIGH",     "assigned_to": "Raj Mehta",   "advance_to": "CORRECTIVE_ACTION"},
    {"product_id": "PROD-003", "title": "API assay out of specification",      "description": "Active ingredient below 98% spec in batch B-2025-91.",               "priority": "CRITICAL", "assigned_to": "Dr. Chen",    "advance_to": "VERIFICATION"},
    {"product_id": "PROD-002", "title": "Outdated SOP used during filling",    "description": "Operator used SOP v2.1 instead of v2.3 during line fill.",           "priority": "MEDIUM",   "assigned_to": "Maria Lopez", "advance_to": "CLOSED"},
    {"product_id": "PROD-004", "title": "Equipment calibration overdue",       "description": "HPLC calibration certificate expired 14 days ago.",                  "priority": "MEDIUM",   "assigned_to": "Tim Nguyen",  "advance_to": None},
    {"product_id": "PROD-003", "title": "Particulate matter in vials",         "description": "Visual inspection found particulates in 12 vials from lot VL-009.",  "priority": "CRITICAL", "assigned_to": "Dr. Patel",   "advance_to": "INVESTIGATION"},
]

_TRANSITION_MAP = {
    "INVESTIGATION":     transition_to_investigation,
    "CORRECTIVE_ACTION": transition_to_corrective_action,
    "VERIFICATION":      transition_to_verification,
    "CLOSED":            close_capa_case,
}

print("Seeding CAPA cases …")
for c in CASES:
    advance_to = c.pop("advance_to")
    result = create_capa_case(**c)
    cid = result["id"]
    print(f"  Created CAPA-{cid:03d}: {c['title'][:50]}")
    if advance_to:
        stages = ["INVESTIGATION", "CORRECTIVE_ACTION", "VERIFICATION", "CLOSED"]
        for stage in stages:
            fn = _TRANSITION_MAP[stage]
            fn(cid)
            print(f"    → advanced to {stage}")
            if stage == advance_to:
                break

print("Done! Board is seeded across all 5 stages.")
