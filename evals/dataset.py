"""
Golden evaluation dataset.

Each case pairs a research topic with a fixed set of facts (and, for offline
scoring, a reference report produced by the pipeline). Fixing the facts makes
the Editor stage reproducible: the deterministic metrics (citation coverage,
structure, source grounding) score the same inputs every run, with no API key
and no live web calls.

When run in ``--live`` mode with an API key, the harness regenerates each
report from these facts and scores *that* instead, so you can measure a real
before/after when you change prompts, models, or retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.models import Fact, Relevance


@dataclass
class EvalCase:
    case_id: str
    topic: str
    sub_queries: list[str]
    facts: list[Fact]
    reference_report: str  # a captured pipeline output, for offline scoring


# A compact, hand-built report fixture used for offline scoring. It intentionally
# cites one source that is NOT in the fact set ("AutoDriveBlog") so the citation
# metric has something to catch — a healthy eval has failing cases.
_BEVFUSION_REPORT = """# BEVFusion vs PointPillars for 3D Object Detection

## Executive Summary
BEVFusion and PointPillars represent two points on the accuracy/latency curve for
3D object detection. BEVFusion fuses camera and LiDAR in a shared bird's-eye-view
space for higher accuracy, while PointPillars remains a fast LiDAR-only baseline [nuScenes].

## Background
3D object detection is core to AV perception; the choice of architecture trades
accuracy against real-time compute budgets.

## Key Findings
- BEVFusion reports **70.2 NDS** on nuScenes, outperforming LiDAR-only baselines [BEVFusion paper].
- PointPillars runs at **>60 FPS**, favouring embedded deployment [PointPillars paper].
- Camera-LiDAR fusion improves detection of small/occluded objects [nuScenes].
- Industry teams report easier deployment of PointPillars on edge GPUs [AutoDriveBlog].

## Architecture Comparison
| Model | Modality | nuScenes NDS | Throughput |
|-------|----------|--------------|------------|
| BEVFusion | Camera + LiDAR | 70.2 | moderate |
| PointPillars | LiDAR | lower | >60 FPS |

## Challenges & Limitations
Fusion models demand calibration and more compute; LiDAR-only models miss
semantic cues from images.

## Conclusion & Outlook
Pick BEVFusion for accuracy-critical ODDs and PointPillars for tight latency
budgets; hybrid distillation is a promising middle ground.

## Sources
1. BEVFusion paper
2. PointPillars paper
3. nuScenes
"""

_ODD_REPORT = """# ODD Exit Detection Strategies in Autonomous Vehicles

## Executive Summary
Operational Design Domain (ODD) exit detection determines when an AV must hand
control back or reach a minimal-risk condition. Approaches range from rule-based
geofencing to learned scene classifiers [SAE J3016].

## Background
An ODD defines where an automated system is designed to operate; detecting its
boundary is safety-critical under ISO 26262 and SOTIF.

## Key Findings
- Geofencing with HD maps gives deterministic ODD boundaries [HD map survey].
- Learned classifiers detect weather/lighting ODD exits with **~92% recall** [perception study].
- Minimal-risk manoeuvres are triggered on confirmed ODD exit [SAE J3016].

## Challenges & Limitations
Learned detectors can be over-confident at distribution edges; map-based methods
go stale without fresh surveys.

## Conclusion & Outlook
Robust systems combine map priors with learned confidence estimation.

## Sources
1. SAE J3016
2. HD map survey
3. perception study
"""


DATASET: list[EvalCase] = [
    EvalCase(
        case_id="bevfusion_vs_pointpillars",
        topic="BEVFusion vs PointPillars for 3D object detection",
        sub_queries=[
            "BEVFusion architecture and camera-LiDAR fusion approach",
            "PointPillars inference speed and embedded deployment",
            "nuScenes NDS/mAP benchmark comparison of the two models",
            "Deployment trade-offs: accuracy vs latency for AV perception",
        ],
        facts=[
            Fact(title="BEVFusion nuScenes accuracy",
                 content="BEVFusion fuses camera and LiDAR in a shared BEV space and reports 70.2 NDS on nuScenes.",
                 source="BEVFusion paper", relevance=Relevance.HIGH),
            Fact(title="PointPillars throughput",
                 content="PointPillars is a LiDAR-only detector that runs at over 60 FPS, suiting embedded GPUs.",
                 source="PointPillars paper", relevance=Relevance.HIGH),
            Fact(title="Fusion helps occluded objects",
                 content="Camera-LiDAR fusion improves detection of small and occluded objects versus LiDAR-only.",
                 source="nuScenes", relevance=Relevance.MEDIUM),
        ],
        reference_report=_BEVFUSION_REPORT,
    ),
    EvalCase(
        case_id="odd_exit_detection",
        topic="ODD exit detection strategies in autonomous vehicles",
        sub_queries=[
            "Geofencing and HD maps for ODD boundary definition",
            "Learned classifiers for weather and lighting ODD exits",
            "Minimal-risk manoeuvre triggers on ODD exit",
            "ISO 26262 / SOTIF considerations for ODD monitoring",
        ],
        facts=[
            Fact(title="Geofencing ODD boundaries",
                 content="HD-map geofencing gives deterministic ODD boundaries for automated driving.",
                 source="HD map survey", relevance=Relevance.HIGH),
            Fact(title="Learned ODD-exit recall",
                 content="Learned classifiers detect weather and lighting ODD exits with around 92% recall.",
                 source="perception study", relevance=Relevance.HIGH),
            Fact(title="Minimal-risk manoeuvre",
                 content="A confirmed ODD exit triggers a minimal-risk manoeuvre per SAE J3016 levels.",
                 source="SAE J3016", relevance=Relevance.MEDIUM),
        ],
        reference_report=_ODD_REPORT,
    ),
]
