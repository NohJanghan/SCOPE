"""AEQA uses the SCOPE scene graph without requiring HM3D semantic assets."""

from src.scene_goatbench import Scene as GoatBenchScene


class Scene(GoatBenchScene):
    use_semantic = False

__all__ = ["Scene"]
