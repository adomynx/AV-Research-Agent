"""
Triage Agent — Analyzes and decomposes a research topic into focused sub-queries
tailored to autonomous driving and perception system domains.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class TriageAgent:
    """
    The Triage Agent is the first stage of the research pipeline.
    It takes a broad research topic and decomposes it into 4-6 focused
    sub-queries that span perception, planning, safety, and benchmarking.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.3
        )
        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research planning specialist for autonomous driving 
and perception systems. Your job is to decompose a research topic into focused, 
searchable sub-queries.

DOMAIN EXPERTISE:
- Perception: object detection, semantic segmentation, 3D reconstruction, sensor fusion
- Sensors: LiDAR, camera, radar, ultrasonic, IMU
- Architectures: YOLO, PointPillars, BEVFusion, Vision Transformers, RT-DETR
- Frameworks: ROS 2, CARLA, AUTOSAR, dSPACE
- Safety: ODD, V&V, ISO 26262, SOTIF
- Datasets: KITTI, nuScenes, Waymo Open, COCO, Cityscapes

Given a topic, generate 4-6 sub-queries that cover different angles:
1. Core technical approach / architecture
2. Benchmarks, datasets, and quantitative comparisons
3. Real-world deployment challenges and edge cases
4. Recent advances and state-of-the-art (2023-2025)
5. Safety validation and regulatory considerations (if relevant)
6. Integration with broader AV stack (if relevant)

Return ONLY a JSON array of strings. No explanation.
Example: ["query 1", "query 2", "query 3", "query 4"]"""),
            ("human", "Research topic: {topic}")
        ])

        self.chain = self.prompt | self.llm | self.parser

    def decompose(self, topic: str) -> list[str]:
        """
        Decompose a research topic into focused sub-queries.

        Args:
            topic: The broad research topic to analyze

        Returns:
            List of 4-6 focused search queries
        """
        try:
            result = self.chain.invoke({"topic": topic})

            if isinstance(result, list):
                return result[:6]

            return [topic]

        except Exception as e:
            print(f"[TriageAgent] Error: {e}")
            # Fallback: return basic decomposition
            return [
                f"{topic} overview and key architectures",
                f"{topic} benchmark results and datasets",
                f"{topic} recent advances 2024 2025",
                f"{topic} deployment challenges and limitations"
            ]
