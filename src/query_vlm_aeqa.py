import logging
from typing import Tuple, Optional, Union

import numpy as np

from src.eval_utils_gpt_aeqa import explore_step
from src.tsdf_planner import TSDFPlanner, SnapShot, Frontier
from src.scene_aeqa import Scene


def query_vlm_for_response(
    question: str,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    verbose: bool = False,
    potential_graph=None,
) -> Optional[Tuple[Union[SnapShot, Frontier], str, int]]:
    step_dict = {
        "obj_map": {
            obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
        },
        "snapshot_objects": {},
        "snapshot_imgs": {},
        "frontier_imgs": [frontier.feature for frontier in tsdf_planner.frontiers],
        "frontier_potential_scores": [],
        "question": question,
    }

    for rgb_id, snapshot in scene.snapshots.items():
        step_dict["snapshot_objects"][rgb_id] = snapshot.cluster
        step_dict["snapshot_imgs"][rgb_id] = scene.all_observations[rgb_id]

    for frontier in tsdf_planner.frontiers:
        if potential_graph is None:
            step_dict["frontier_potential_scores"].append(3.0)
            continue
        try:
            frontier_world_pos = potential_graph._voxel_to_world(frontier.position)
            potential_position = np.array([frontier_world_pos[0], frontier_world_pos[2]])
            step_dict["frontier_potential_scores"].append(
                potential_graph.get_potential_at_position(potential_position)
            )
        except Exception:
            logging.exception("Failed to read frontier potential score")
            step_dict["frontier_potential_scores"].append(3.0)

    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    if not step_dict["snapshot_imgs"] and not step_dict["frontier_imgs"]:
        logging.error("No snapshots or frontiers available for VLM query")
        return None

    outputs, snapshot_id_mapping, reason, n_filtered_snapshots = explore_step(
        step_dict, cfg, verbose=verbose
    )
    if outputs is None:
        logging.error("explore_step failed and returned None")
        return None

    response_parts = outputs.split()
    if len(response_parts) < 2:
        logging.error(f"Wrong output format: {outputs}")
        return None

    target_type, target_index = response_parts[:2]
    if target_type not in ("snapshot", "frontier") or not target_index.isdigit():
        logging.error(f"Wrong target format: {outputs}")
        return None

    target_index = int(target_index)
    if target_type == "snapshot":
        if not 0 <= target_index < len(snapshot_id_mapping):
            logging.error(f"Snapshot index out of range: {target_index}")
            return None
        target_index = snapshot_id_mapping[target_index]
        if not 0 <= target_index < len(scene.snapshots):
            logging.error(f"Mapped snapshot index out of range: {target_index}")
            return None
        pred_target_snapshot = list(scene.snapshots.values())[target_index]
        logging.info(f"Next choice: Snapshot of {pred_target_snapshot.image}")
        return pred_target_snapshot, reason, n_filtered_snapshots

    if target_index >= len(tsdf_planner.frontiers):
        logging.error(f"Frontier index out of range: {target_index}")
        return None
    pred_target_frontier = tsdf_planner.frontiers[target_index]
    logging.info(f"Next choice: Frontier at {pred_target_frontier.position}")
    return pred_target_frontier, reason, n_filtered_snapshots
