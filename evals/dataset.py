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
    EvalCase(
        case_id="lidar_camera_fusion",
        topic="LiDAR-camera fusion architectures for AV perception",
        sub_queries=[
            "Query/attention-based fusion (TransFusion) vs BEV-level fusion",
            "nuScenes accuracy of camera-LiDAR fusion detectors",
            "Robustness of fusion to sensor/modality dropout",
        ],
        facts=[
            Fact(title="TransFusion attention fusion",
                 content="TransFusion uses a transformer decoder to fuse LiDAR BEV features with image features via cross-attention.",
                 source="TransFusion paper", relevance=Relevance.HIGH),
            Fact(title="Fusion nuScenes gain",
                 content="Camera-LiDAR fusion detectors improve nuScenes NDS over LiDAR-only baselines.",
                 source="nuScenes", relevance=Relevance.HIGH),
            Fact(title="Modality dropout robustness",
                 content="DeepFusion-style methods degrade gracefully when one modality is missing or corrupted.",
                 source="DeepFusion paper", relevance=Relevance.MEDIUM),
        ],
        reference_report="""# LiDAR-Camera Fusion Architectures for AV Perception

## Executive Summary
Camera-LiDAR fusion combines geometric precision with semantic richness, with
query/attention designs like TransFusion leading recent benchmarks [TransFusion paper].

## Key Findings
- TransFusion fuses modalities via cross-attention in a transformer decoder [TransFusion paper].
- Fusion improves nuScenes NDS over LiDAR-only baselines [nuScenes].
- Attention fusion degrades gracefully under modality dropout [DeepFusion paper].

## Challenges & Limitations
Fusion adds calibration and compute cost; sensitivity to cross-sensor misalignment remains.

## Conclusion & Outlook
Attention-based fusion is the current direction; robustness under corruption is the open frontier.

## Sources
1. TransFusion paper
2. nuScenes
3. DeepFusion paper
""",
    ),
    EvalCase(
        case_id="transformer_perception",
        topic="Transformer architectures in autonomous driving perception",
        sub_queries=[
            "DETR3D query-based 3D detection approach",
            "RT-DETR real-time detection trade-offs",
            "Transformer vs CNN detector accuracy on AV datasets",
        ],
        facts=[
            Fact(title="DETR3D queries",
                 content="DETR3D uses 3D object queries that sample multi-view image features without dense BEV construction.",
                 source="DETR3D paper", relevance=Relevance.HIGH),
            Fact(title="RT-DETR real-time",
                 content="RT-DETR removes NMS and achieves real-time detection with competitive accuracy.",
                 source="RT-DETR paper", relevance=Relevance.HIGH),
        ],
        reference_report="""# Transformer Architectures in AV Perception

## Executive Summary
Transformer detectors replace hand-tuned anchors and NMS with learned queries,
with DETR3D and RT-DETR representing 3D and real-time variants [DETR3D paper].

## Key Findings
- DETR3D samples multi-view features with 3D object queries, avoiding dense BEV [DETR3D paper].
- RT-DETR removes NMS for real-time, competitive detection [RT-DETR paper].

## Challenges & Limitations
Transformer detectors can be data- and compute-hungry and slower to converge.

## Conclusion & Outlook
Query-based detection is maturing toward real-time AV deployment.

## Sources
1. DETR3D paper
2. RT-DETR paper
""",
    ),
    EvalCase(
        case_id="edge_deployment",
        topic="Real-time object detection on edge devices for robotics",
        sub_queries=[
            "TensorRT INT8 quantization accuracy/latency trade-offs",
            "Jetson throughput for detection models",
            "YOLO-family suitability for embedded inference",
        ],
        facts=[
            Fact(title="TensorRT INT8",
                 content="TensorRT INT8 quantization reduces latency with a small accuracy drop versus FP16.",
                 source="NVIDIA TensorRT docs", relevance=Relevance.HIGH),
            Fact(title="Jetson throughput",
                 content="Jetson-class GPUs run compact detectors in real time at tens of FPS.",
                 source="Jetson benchmark", relevance=Relevance.MEDIUM),
            Fact(title="YOLO efficiency",
                 content="YOLO-family single-stage detectors favour embedded real-time inference.",
                 source="YOLO paper", relevance=Relevance.HIGH),
        ],
        reference_report="""# Real-Time Detection on Edge Devices

## Executive Summary
Edge detection balances accuracy against tight latency and power budgets, using
quantization and efficient single-stage detectors [NVIDIA TensorRT docs].

## Key Findings
- TensorRT INT8 cuts latency with a small accuracy drop vs FP16 [NVIDIA TensorRT docs].
- Jetson-class GPUs run compact detectors at tens of FPS [Jetson benchmark].
- YOLO-family detectors favour embedded real-time inference [YOLO paper].

## Challenges & Limitations
Quantization can hurt small-object accuracy; thermal and power limits constrain throughput.

## Conclusion & Outlook
INT8 single-stage detectors are the practical default for robotics edge perception.

## Sources
1. NVIDIA TensorRT docs
2. Jetson benchmark
3. YOLO paper
""",
    ),
    EvalCase(
        case_id="sotif_validation",
        topic="Safety validation methods for AV perception systems",
        sub_queries=[
            "SOTIF (ISO 21448) approach to perception safety",
            "Scenario-based validation and coverage",
            "Operational experience from AV safety reports",
        ],
        facts=[
            Fact(title="SOTIF scope",
                 content="ISO 21448 (SOTIF) addresses hazards from functional insufficiencies without a component fault.",
                 source="ISO 21448", relevance=Relevance.HIGH),
            Fact(title="Scenario coverage",
                 content="Scenario-based testing targets known and edge-case conditions to bound residual risk.",
                 source="SOTIF survey", relevance=Relevance.MEDIUM),
        ],
        reference_report="""# Safety Validation Methods for AV Perception

## Executive Summary
Perception safety relies on SOTIF (ISO 21448) to manage functional insufficiencies,
combined with scenario-based validation [ISO 21448].

## Key Findings
- SOTIF addresses hazards from insufficiencies without a component fault [ISO 21448].
- Scenario-based testing bounds residual risk across edge cases [SOTIF survey].

## Challenges & Limitations
Completeness of the scenario space is unprovable; rare long-tail events dominate residual risk.

## Conclusion & Outlook
SOTIF plus scenario coverage and field monitoring is the accepted validation backbone.

## Sources
1. ISO 21448
2. SOTIF survey
""",
    ),
    EvalCase(
        case_id="radar_camera_fusion",
        topic="Radar-camera fusion for robust AV perception",
        sub_queries=[
            "Radar-camera association and fusion methods",
            "Radar advantages in adverse weather",
            "nuScenes radar fusion benchmark results",
        ],
        facts=[
            Fact(title="CRAFT association",
                 content="CRAFT performs radar-camera cross-attention to associate sparse radar points with image regions.",
                 source="CRAFT paper", relevance=Relevance.HIGH),
            Fact(title="Radar weather robustness",
                 content="Radar is robust to rain, fog, and low light where cameras degrade.",
                 source="radar survey", relevance=Relevance.HIGH),
        ],
        reference_report="""# Radar-Camera Fusion for Robust AV Perception

## Executive Summary
Radar-camera fusion adds all-weather robustness to semantic cameras, with
cross-attention association methods like CRAFT [CRAFT paper].

## Key Findings
- CRAFT associates sparse radar points with image regions via cross-attention [CRAFT paper].
- Radar stays robust in rain, fog, and low light where cameras degrade [radar survey].

## Challenges & Limitations
Radar sparsity and angular resolution limit fine localization; association remains hard.

## Conclusion & Outlook
Radar-camera fusion is key to all-weather perception; richer 4D radar is a promising direction.

## Sources
1. CRAFT paper
2. radar survey
""",
    ),
]
