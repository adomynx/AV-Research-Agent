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


def _report(title: str, summary: str, findings: list[str],
            limitations: str, conclusion: str, sources: list[str]) -> str:
    """Build a structured fixture report with all required sections."""
    fl = "\n".join(f"- {f}" for f in findings)
    sr = "\n".join(f"{i}. {s}" for i, s in enumerate(sources, 1))
    return (
        f"# {title}\n\n"
        f"## Executive Summary\n{summary}\n\n"
        f"## Key Findings\n{fl}\n\n"
        f"## Challenges & Limitations\n{limitations}\n\n"
        f"## Conclusion & Outlook\n{conclusion}\n\n"
        f"## Sources\n{sr}\n"
    )


def _case(case_id: str, topic: str, sub_queries: list[str],
          facts_spec: list[tuple], summary: str, findings: list[str],
          limitations: str, conclusion: str) -> "EvalCase":
    """Compact EvalCase builder. facts_spec: list of (title, content, source, relevance)."""
    facts = [Fact(title=t, content=c, source=s, relevance=r) for t, c, s, r in facts_spec]
    sources = list(dict.fromkeys(f.source for f in facts))
    report = _report(topic, summary, findings, limitations, conclusion, sources)
    return EvalCase(case_id=case_id, topic=topic, sub_queries=sub_queries,
                    facts=facts, reference_report=report)


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
    _case(
        "voxelnet_second",
        "Voxel-based 3D object detection (VoxelNet and SECOND)",
        ["VoxelNet voxel feature encoding", "SECOND sparse convolutions", "Voxel resolution trade-offs"],
        [("VoxelNet voxel features", "VoxelNet learns voxel-wise features with 3D convolutions for LiDAR detection.", "VoxelNet paper", Relevance.HIGH),
         ("SECOND sparse conv", "SECOND uses sparse 3D convolutions to make voxel detection fast and practical.", "SECOND paper", Relevance.HIGH)],
        "Voxel-based detectors grid point clouds; SECOND's sparse convolutions made them practical [SECOND paper].",
        ["VoxelNet learns voxel features via 3D convolutions [VoxelNet paper].",
         "SECOND accelerates this with sparse 3D convolutions [SECOND paper]."],
        "Voxelization loses fine geometry; resolution trades accuracy against compute.",
        "Sparse-convolution voxel backbones remain a strong LiDAR detection foundation.",
    ),
    _case(
        "centerpoint",
        "Center-based 3D detection and tracking (CenterPoint)",
        ["CenterPoint center representation", "nuScenes detection/tracking gains", "Anchor-free advantages"],
        [("CenterPoint centers", "CenterPoint detects objects as center points and regresses attributes, easing orientation handling.", "CenterPoint paper", Relevance.HIGH),
         ("nuScenes gains", "CenterPoint improves nuScenes detection and tracking over anchor-based methods.", "nuScenes", Relevance.HIGH)],
        "CenterPoint reframes 3D detection as keypoint estimation, simplifying orientation and tracking [CenterPoint paper].",
        ["Objects are detected as centers, avoiding anchor design [CenterPoint paper].",
         "It improves nuScenes detection and tracking metrics [nuScenes]."],
        "Center heatmaps can struggle with very dense scenes and small objects.",
        "Center-based detection is a widely adopted, simple and strong baseline.",
    ),
    _case(
        "occupancy_prediction",
        "3D occupancy prediction for AV perception",
        ["Dense occupancy vs bounding boxes", "Camera-only occupancy", "Arbitrary-shape obstacles"],
        [("Dense occupancy", "Occupancy networks predict dense 3D occupancy instead of boxes, capturing arbitrary shapes.", "Occ3D paper", Relevance.HIGH),
         ("Vision occupancy", "Camera-only occupancy prediction is competitive for general obstacle detection.", "TPVFormer paper", Relevance.MEDIUM)],
        "Occupancy prediction captures arbitrary-shaped obstacles that box detectors miss [Occ3D paper].",
        ["Dense 3D occupancy represents arbitrary shapes beyond boxes [Occ3D paper].",
         "Camera-only occupancy is competitive for general obstacles [TPVFormer paper]."],
        "Dense voxel outputs are compute- and memory-heavy at high resolution.",
        "Occupancy is emerging as a general representation for open-set obstacles.",
    ),
    _case(
        "monocular_3d",
        "Monocular 3D object detection for AVs",
        ["Depth ambiguity in monocular 3D", "Pseudo-LiDAR approach", "Accuracy gap vs LiDAR"],
        [("Depth ambiguity", "Monocular 3D detection infers depth from a single image, an ill-posed problem.", "FCOS3D paper", Relevance.HIGH),
         ("Pseudo-LiDAR", "Pseudo-LiDAR lifts monocular depth into point clouds to reuse LiDAR detectors.", "Pseudo-LiDAR paper", Relevance.MEDIUM)],
        "Monocular 3D detection is cheap but depth-ambiguous; pseudo-LiDAR narrows the gap [FCOS3D paper].",
        ["Single-image depth makes monocular 3D ill-posed [FCOS3D paper].",
         "Pseudo-LiDAR converts depth to points to reuse LiDAR detectors [Pseudo-LiDAR paper]."],
        "Depth errors dominate accuracy, especially for distant objects.",
        "Monocular 3D suits cost-sensitive deployments where some accuracy loss is acceptable.",
    ),
    _case(
        "multi_object_tracking",
        "3D multi-object tracking for autonomous driving",
        ["Tracking-by-detection pipelines", "Motion vs appearance association", "nuScenes AMOTA metric"],
        [("Tracking-by-detection", "Most AV MOT uses tracking-by-detection with motion and appearance association.", "AB3DMOT paper", Relevance.HIGH),
         ("AMOTA metric", "3D tracking quality is measured by AMOTA/AMOTP on nuScenes.", "nuScenes", Relevance.MEDIUM)],
        "AV tracking is dominated by tracking-by-detection with motion-based association [AB3DMOT paper].",
        ["Tracking-by-detection associates detections across frames [AB3DMOT paper].",
         "AMOTA/AMOTP on nuScenes are standard 3D tracking metrics [nuScenes]."],
        "Association degrades under occlusion, missed detections, and ID switches.",
        "Simple motion-based 3D trackers are strong; learned association is the frontier.",
    ),
    _case(
        "hd_maps_localization",
        "HD maps and localization for autonomous driving",
        ["HD map lane-level priors", "LiDAR-to-map matching", "Map freshness challenges"],
        [("HD map priors", "HD maps provide lane-level priors that constrain localization and planning.", "HD map survey", Relevance.HIGH),
         ("Map matching", "LiDAR-to-map matching yields centimeter-level localization in mapped areas.", "localization study", Relevance.MEDIUM)],
        "HD maps give lane-level priors enabling centimeter localization via map matching [HD map survey].",
        ["HD maps constrain localization and planning with lane-level priors [HD map survey].",
         "LiDAR-to-map matching achieves centimeter localization [localization study]."],
        "Maps go stale without resurveying; unmapped areas lose the prior.",
        "Map priors plus online matching remain central; mapless driving is an active goal.",
    ),
    _case(
        "trajectory_prediction",
        "Trajectory prediction for autonomous driving",
        ["Multimodal future prediction", "Interaction modeling", "Argoverse forecasting metrics"],
        [("Multimodal futures", "Trajectory models output multimodal futures with uncertainty.", "Trajectron++ paper", Relevance.HIGH),
         ("Argoverse benchmark", "Argoverse benchmarks motion forecasting with minADE/minFDE.", "Argoverse", Relevance.MEDIUM)],
        "Prediction models output multimodal, uncertain futures, benchmarked on Argoverse [Trajectron++ paper].",
        ["Models predict multimodal futures with uncertainty [Trajectron++ paper].",
         "Argoverse measures forecasting via minADE/minFDE [Argoverse]."],
        "Rare interactive maneuvers and long horizons remain hard to predict.",
        "Graph and transformer predictors with interaction modeling lead the field.",
    ),
    _case(
        "end_to_end_driving",
        "End-to-end planning-oriented autonomous driving (UniAD)",
        ["Unified perception-prediction-planning", "Planning-oriented design", "Open-loop planning metrics"],
        [("UniAD unified", "UniAD unifies perception, prediction, and planning in one end-to-end network.", "UniAD paper", Relevance.HIGH),
         ("Open-loop metrics", "End-to-end methods are evaluated on open-loop planning metrics on nuScenes.", "nuScenes", Relevance.MEDIUM)],
        "UniAD unifies the stack into one planning-oriented end-to-end network [UniAD paper].",
        ["UniAD jointly learns perception, prediction, and planning [UniAD paper].",
         "End-to-end planning is assessed with open-loop metrics [nuScenes]."],
        "Open-loop metrics can overstate closed-loop driving performance.",
        "Planning-oriented end-to-end design is a major research direction.",
    ),
    _case(
        "sim2real_adaptation",
        "Sim-to-real domain adaptation for AV perception",
        ["Sim-to-real domain gap", "Domain randomization", "Adaptation techniques"],
        [("Domain gap", "Sim-to-real transfer suffers a domain gap in appearance and sensor noise.", "CARLA", Relevance.HIGH),
         ("Domain adaptation", "Domain adaptation and randomization narrow the sim-to-real gap.", "domain adaptation survey", Relevance.MEDIUM)],
        "Simulators like CARLA train perception cheaply but face a sim-to-real domain gap [CARLA].",
        ["A domain gap in appearance and sensor noise hurts transfer [CARLA].",
         "Randomization and adaptation narrow the gap [domain adaptation survey]."],
        "Residual gaps persist; validating real-world safety from sim remains hard.",
        "Sim plus adaptation scales data; closing the last gap is the open problem.",
    ),
    _case(
        "adverse_weather",
        "Perception robustness in adverse weather",
        ["Rain/fog/snow sensor degradation", "Radar and gated imaging", "Weather-robust benchmarks"],
        [("Weather degradation", "Rain, fog, and snow degrade camera and LiDAR returns.", "adverse weather survey", Relevance.HIGH),
         ("Robust sensing", "Radar and gated imaging improve robustness in adverse weather.", "radar survey", Relevance.MEDIUM)],
        "Adverse weather degrades optical sensors; radar and gated imaging add robustness [adverse weather survey].",
        ["Rain, fog, and snow degrade camera and LiDAR [adverse weather survey].",
         "Radar and gated imaging are more weather-robust [radar survey]."],
        "Labeled adverse-weather data is scarce and long-tailed.",
        "Multi-modal, weather-robust sensing is required for all-conditions autonomy.",
    ),
    _case(
        "pedestrian_detection",
        "Pedestrian detection for autonomous vehicles",
        ["Small and occluded pedestrians", "Crowded-scene benchmarks", "Safety-critical recall"],
        [("Hard instances", "Pedestrian detection is hard for small, distant, and occluded instances.", "Caltech Pedestrian", Relevance.HIGH),
         ("Crowd benchmark", "CrowdHuman benchmarks detection under heavy occlusion.", "CrowdHuman", Relevance.MEDIUM)],
        "Pedestrian detection is safety-critical and hardest for small, occluded instances [Caltech Pedestrian].",
        ["Small, distant, occluded pedestrians are the hard cases [Caltech Pedestrian].",
         "CrowdHuman stresses detection under heavy occlusion [CrowdHuman]."],
        "High recall is required for safety, raising false-positive pressure.",
        "Occlusion-robust, high-recall detection remains a priority for AV safety.",
    ),
    _case(
        "nerf_reconstruction",
        "NeRF scene reconstruction for AV simulation",
        ["NeRF photorealistic reconstruction", "Street-scene NeRFs", "Closed-loop sensor simulation"],
        [("NeRF scenes", "NeRF reconstructs photorealistic 3D scenes from posed images for simulation.", "NeRF paper", Relevance.HIGH),
         ("Driving NeRF", "Street-scene NeRFs enable closed-loop sensor simulation for AV testing.", "Block-NeRF paper", Relevance.MEDIUM)],
        "NeRFs reconstruct photorealistic scenes, enabling closed-loop sensor simulation for AVs [NeRF paper].",
        ["NeRF builds photorealistic 3D scenes from images [NeRF paper].",
         "Street-scene NeRFs enable closed-loop simulation [Block-NeRF paper]."],
        "NeRFs are compute-heavy and struggle with dynamic actors and scale.",
        "Neural reconstruction is reshaping AV simulation and data generation.",
    ),
    _case(
        "v2x_cooperative",
        "V2X cooperative perception for autonomous driving",
        ["Cross-agent sensor sharing", "Occlusion recovery", "Bandwidth vs accuracy trade-off"],
        [("V2X sharing", "V2X cooperative perception shares sensor data across vehicles and infrastructure to see occluded objects.", "V2X-Sim paper", Relevance.HIGH),
         ("Bandwidth tradeoff", "Intermediate feature fusion balances accuracy against communication bandwidth.", "cooperative perception survey", Relevance.MEDIUM)],
        "V2X cooperative perception shares data across agents to see around occlusions [V2X-Sim paper].",
        ["Sharing sensor data recovers occluded objects [V2X-Sim paper].",
         "Intermediate feature fusion trades accuracy against bandwidth [cooperative perception survey]."],
        "Latency, bandwidth, and synchronization limit real-world V2X.",
        "Cooperative perception extends the sensing horizon beyond a single vehicle.",
    ),
]
