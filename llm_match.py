import os
import json
import argparse
import math
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile


DATA_DIR = Path(__file__).resolve().parent / "data"


def truncate_prediction(prediction):
    # pre-process answers (from open-eqa/evaluate-predictions.py)
    if prediction:
        end_idx = prediction.rfind(".")
        if end_idx >= 0 and end_idx + 1 < len(prediction):
            prediction = prediction[: end_idx + 1]
    return prediction


def evaluate_llm_match(
    question,
    answer,
    prediction,
    extra_answers,
    model,
    dismiss_error=False,
):
    if "OPENAI_API_KEY" not in os.environ and os.getenv("OPENAI_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
    if "OPENAI_BASE_URL" not in os.environ and os.getenv("END_POINT"):
        os.environ["OPENAI_BASE_URL"] = os.environ["END_POINT"]

    from openeqa.evaluation.llm_match import get_llm_match_score

    prediction = truncate_prediction(prediction)
    try:
        return get_llm_match_score(
            question=question,
            answer=answer,
            prediction=prediction,
            extra_answers=extra_answers,
            openai_model=model,
        )
    except ValueError as e:
        if not dismiss_error:
            raise
        print(e)
        return 1


def normalize_llm_match_score(score):
    score = min(max(float(score), 1.0), 5.0)
    return 100.0 * (score - 1.0) / 4.0


def blind_cache_path(prefix, answer_model):
    if not answer_model:
        raise ValueError("VLM_MODEL must be set for blind-LLM evaluation")
    filename_model = answer_model.replace("/", "_")
    return DATA_DIR / f"{prefix}_{filename_model}.json"


def load_cache(path, metadata, value_type):
    if not path.exists():
        return {**metadata, "data": {}}

    with path.open("r", encoding="utf-8") as f:
        cache = json.load(f)
    if not isinstance(cache, dict) or not isinstance(cache.get("data"), dict):
        raise ValueError(f"Invalid blind-LLM cache structure: {path}")
    for key, expected in metadata.items():
        if cache.get(key) != expected:
            raise ValueError(
                f"Blind-LLM cache metadata mismatch for {path}: "
                f"expected {key}={expected!r}, got {cache.get(key)!r}"
            )
    for question_id, value in cache["data"].items():
        if value_type is str:
            valid = isinstance(value, str) and bool(value.strip())
        else:
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 1.0 <= float(value) <= 5.0
            )
        if not valid:
            raise ValueError(
                f"Invalid blind-LLM cache value for question_id {question_id}: "
                f"{value!r}"
            )
    return cache


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary_path = Path(f.name)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def generate_blind_answer(question, model):
    from openai import OpenAI
    from openeqa.baselines.gpt4 import parse_output
    from openeqa.utils.prompt_utils import load_prompt

    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("END_POINT") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise ValueError(
            "OPENAI_KEY or OPENAI_API_KEY must be set for blind-LLM answers"
        )
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    prompt = load_prompt("blind-llm").format(question=question) + (
        '\nRespond with exactly one line beginning with "A: ".'
    )
    last_error = None
    for attempt in range(3):
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            seed=1234 + attempt,
            max_tokens=128,
            reasoning_effort="none",
        )
        content = completion.choices[0].message.content
        try:
            if not content:
                raise ValueError("Blind-LLM response has no message content")
            return parse_output(content)
        except ValueError as error:
            last_error = error
    raise last_error


def prepare_blind_scores(
    question_id_to_item,
    answer_model,
    judger_model,
    dismiss_error=False,
):
    if not question_id_to_item:
        return {}

    answer_path = blind_cache_path("blind_llm_answer", answer_model)
    score_path = blind_cache_path("blind_llm_score", answer_model)
    answer_cache = load_cache(
        answer_path,
        {"answer_model": answer_model},
        str,
    )
    score_cache = load_cache(
        score_path,
        {"answer_model": answer_model, "judger_model": judger_model},
        float,
    )

    created_answer_ids = set()
    for question_id in sorted(question_id_to_item):
        item = question_id_to_item[question_id]
        if question_id not in answer_cache["data"]:
            answer_cache["data"][question_id] = generate_blind_answer(
                item["question"],
                answer_model,
            )
            created_answer_ids.add(question_id)
            write_json(answer_path, answer_cache)

    for question_id in sorted(question_id_to_item):
        item = question_id_to_item[question_id]
        if (
            question_id in created_answer_ids
            or question_id not in score_cache["data"]
        ):
            score_cache["data"][question_id] = evaluate_llm_match(
                question=item["question"],
                answer=item["answer"],
                prediction=answer_cache["data"][question_id],
                extra_answers=item.get("extra_answers"),
                model=judger_model,
                dismiss_error=dismiss_error,
            )
            write_json(score_path, score_cache)

    return {
        question_id: score_cache["data"][question_id]
        for question_id in question_id_to_item
    }


def load_blind_scores(failed_ids, answer_model, judger_model):
    if not failed_ids:
        return {}

    score_path = blind_cache_path("blind_llm_score", answer_model)
    score_cache = load_cache(
        score_path,
        {"answer_model": answer_model, "judger_model": judger_model},
        float,
    )
    missing_score_ids = sorted(failed_ids - set(score_cache["data"]))
    if missing_score_ids:
        raise ValueError(
            "해결되지 않은 질문에 대한 blind llm 점수가 없다: "
            f"{missing_score_ids}. "
            "먼저 `llm_match.py --prepare`를 실행하세요."
        )
    return {
        question_id: score_cache["data"][question_id]
        for question_id in failed_ids
    }


def compute_spl_coefficient(path_length, gt_path_length):
    path_length = float(path_length)
    gt_path_length = float(gt_path_length)

    if (
        gt_path_length <= 0
        or path_length < 0
        or not math.isfinite(gt_path_length)
        or not math.isfinite(path_length)
    ):
        raise ValueError(
            f"Invalid path lengths: path_length={path_length}, "
            f"gt_path_length={gt_path_length}"
        )

    return gt_path_length / max(gt_path_length, path_length)


def compute_llm_match_spl_score(
    question_id,
    llm_match_score,
    path_length_map,
    gt_path_length_map,
):
    if question_id not in gt_path_length_map:
        raise KeyError(f"Missing GT path length for question_id: {question_id}")
    if question_id not in path_length_map:
        raise KeyError(f"Missing path length for question_id: {question_id}")

    normalized_score = normalize_llm_match_score(llm_match_score)
    return normalized_score * compute_spl_coefficient(
        path_length_map[question_id],
        gt_path_length_map[question_id],
    )


def unpack_result(question_id, result):
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ValueError(
            f"Expected result for question_id {question_id} to be "
            f"(llm_match, llm_match_spl), got: {result}"
        )
    return result


def prediction_id_set(predictions):
    question_ids = [prediction["question_id"] for prediction in predictions]
    unique_ids = set(question_ids)
    if len(unique_ids) != len(question_ids):
        duplicates = sorted(
            question_id
            for question_id in unique_ids
            if question_ids.count(question_id) > 1
        )
        raise ValueError(f"Duplicate prediction question IDs: {duplicates}")
    return unique_ids


def validate_resume_result_ids(results, prediction_ids):
    stale_result_ids = sorted(set(results) - prediction_ids)
    if stale_result_ids:
        raise ValueError(
            "Existing LLM Match scores are missing from the current "
            f"predictions: {stale_result_ids}"
        )


def compute_statistics(
    results,
    total_questions=None,
    total_predictions=None,
    blind_scores=None,
):
    llm_match_scores = []
    llm_match_spl_scores = []
    for question_id, result in results.items():
        llm_match_score, llm_match_spl_score = unpack_result(question_id, result)
        llm_match_scores.append(normalize_llm_match_score(llm_match_score))
        llm_match_spl_score = float(llm_match_spl_score)
        if not math.isfinite(llm_match_spl_score) or not (
            0.0 <= llm_match_spl_score <= 100.0
        ):
            raise ValueError(
                f"Invalid LLM Match SPL score for {question_id}: "
                f"{llm_match_spl_score}"
            )
        llm_match_spl_scores.append(llm_match_spl_score)

    statistics = {
        "completed_only_llm_match_mean": round(
            sum(llm_match_scores) / len(llm_match_scores), 2
        )
        if llm_match_scores
        else 0.0,
        "completed_only_llm_match_spl_mean": round(
            sum(llm_match_spl_scores) / len(llm_match_spl_scores), 2
        )
        if llm_match_spl_scores
        else 0.0,
    }
    if total_questions is not None:
        total_questions = int(total_questions)
        if total_predictions is None:
            total_predictions = len(results)
        total_predictions = int(total_predictions)
        if not len(results) <= total_predictions <= total_questions:
            raise ValueError(
                "Expected len(results) <= total_predictions <= total_questions"
            )
        statistics.update(
            {
                "prediction_coverage": round(
                    100.0 * total_predictions / total_questions, 2
                )
                if total_questions
                else 0.0,
                "evaluation_coverage": round(
                    100.0 * len(results) / total_predictions, 2
                )
                if total_predictions
                else 100.0,
            }
        )
        if blind_scores is not None:
            if total_predictions + len(blind_scores) != total_questions:
                raise ValueError(
                    "Expected completed predictions plus failed episodes to equal "
                    "the number of dataset questions"
                )
        if len(results) == total_predictions and blind_scores is not None:
            normalized_blind_scores = [
                normalize_llm_match_score(score) for score in blind_scores.values()
            ]
            statistics.update(
                {
                    "original_llm_match_mean": round(
                        (
                            sum(llm_match_scores)
                            + sum(normalized_blind_scores)
                        )
                        / total_questions,
                        2,
                    )
                    if total_questions
                    else 0.0,
                    # Failed episodes use blind answers for LLM Match but have
                    # zero SPL contribution, following the 3D-Mem A-EQA setup.
                    "original_llm_match_spl_mean": round(
                        sum(llm_match_spl_scores) / total_questions, 2
                    )
                    if total_questions
                    else 0.0,
                    "grounding_llm_match_mean": round(
                        sum(llm_match_scores) / total_questions, 2
                    )
                    if total_questions
                    else 0.0,
                    "grounding_llm_match_spl_mean": round(
                        sum(llm_match_spl_scores) / total_questions, 2
                    )
                    if total_questions
                    else 0.0,
                }
            )
    return statistics


def write_results(output_path, results, statistics=None):
    output_data = {
        "results": results,
        "statistic": statistics if statistics is not None else {},
        "number of loaded predictions": len(results),
    }
    write_json(output_path, output_data)


def prompt_interactive_inputs():
    question = input("Question: ")
    answer = input("Answer: ")

    print("Extra answers (optional, one per line; empty line to finish):")
    extra_answers = []
    while True:
        extra_answer = input()
        if not extra_answer:
            break
        extra_answers.append(extra_answer)

    prediction = input("Prediction: ")
    return question, answer, extra_answers or None, prediction


def interactive_main(config, dismiss_error=False):
    question, answer, extra_answers, prediction = prompt_interactive_inputs()
    score = evaluate_llm_match(
        question=question,
        answer=answer,
        prediction=prediction,
        extra_answers=extra_answers,
        model=config.model,
        dismiss_error=dismiss_error,
    )
    print(f"LLM match score: {score}")


def prepare_main(config, dismiss_error=False):
    dataset = json.load(open(config.dataset, "r"))
    dataset_question_ids = [item["question_id"] for item in dataset]
    if len(set(dataset_question_ids)) != len(dataset_question_ids):
        raise ValueError("Dataset contains duplicate question IDs")
    question_id_to_item = {item["question_id"]: item for item in dataset}
    print(f"Loaded {len(dataset):,} ground truth questions")

    blind_scores = prepare_blind_scores(
        question_id_to_item=question_id_to_item,
        answer_model=os.getenv("VLM_MODEL"),
        judger_model=config.model,
        dismiss_error=dismiss_error,
    )
    print(f"Prepared {len(blind_scores):,} blind-LLM scores")


def main(config, dismiss_error=False):
    from tqdm import tqdm

    dataset = json.load(open(config.dataset, "r"))
    dataset_question_ids = [item["question_id"] for item in dataset]
    if len(set(dataset_question_ids)) != len(dataset_question_ids):
        raise ValueError("Dataset contains duplicate question IDs")
    question_id_to_item = {item["question_id"]: item for item in dataset}
    print(f"Loaded {len(dataset):,} ground truth questions")

    gpt_answer_path = os.path.join(config.output_dir, config.gpt_answer)
    predictions = json.load(open(gpt_answer_path, "r"))
    print(f"Loaded {len(predictions):,} predictions")
    prediction_ids = prediction_id_set(predictions)

    fail_list_path = os.path.join(
        config.output_dir,
        config.get("fail_list", "fail_list.pkl"),
    )
    if not os.path.exists(fail_list_path):
        raise FileNotFoundError(f"Failure list file not found: {fail_list_path}")
    with open(fail_list_path, "rb") as f:
        failed_id_list = pickle.load(f)
    if not isinstance(failed_id_list, list) or any(
        not isinstance(question_id, str) for question_id in failed_id_list
    ):
        raise ValueError("Failure list must be a list of string question IDs")
    failed_ids = set(failed_id_list)
    if len(failed_ids) != len(failed_id_list):
        raise ValueError("Failure list contains duplicate question IDs")

    dataset_ids = set(question_id_to_item)
    overlap = prediction_ids & failed_ids
    if overlap:
        raise ValueError(
            f"Question IDs appear in both predictions and failure list: {sorted(overlap)}"
        )
    unknown_outcome_ids = (prediction_ids | failed_ids) - dataset_ids
    if unknown_outcome_ids:
        raise KeyError(
            f"Episode outcomes are missing from the dataset: {sorted(unknown_outcome_ids)}"
        )
    missing_outcome_ids = dataset_ids - prediction_ids - failed_ids
    if missing_outcome_ids:
        raise ValueError(
            "Cannot report full metrics because some dataset questions have neither "
            f"a prediction nor a failed outcome: {sorted(missing_outcome_ids)}"
        )

    blind_scores = load_blind_scores(
        failed_ids=failed_ids,
        answer_model=os.getenv("VLM_MODEL"),
        judger_model=config.model,
    )
    print(f"Loaded {len(blind_scores):,} blind-LLM scores for failed episodes")

    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, config.output)
    path_length_path = os.path.join(config.output_dir, config.path_length)
    gt_path_length_path = config.get("gt_path_length", "data/gt_path_length.json")
    if not path_length_path or not os.path.exists(path_length_path):
        raise FileNotFoundError(f"Path length file not found: {path_length_path}")
    with open(path_length_path, "rb") as f:
        path_length_map = pickle.load(f)

    if not gt_path_length_path or not os.path.exists(gt_path_length_path):
        raise FileNotFoundError(f"GT path length file not found: {gt_path_length_path}")
    with open(gt_path_length_path, "r") as f:
        gt_path_length_map = json.load(f)

    results = {}
    if os.path.exists(output_path):
        existing = json.load(open(output_path, "r"))
        results = existing["results"]
        validate_resume_result_ids(results, prediction_ids)
        for question_id, result in results.items():
            if question_id not in question_id_to_item:
                raise KeyError(f"question_id {question_id} not found in dataset")
            if question_id not in gt_path_length_map:
                raise KeyError(f"Missing GT path length for question_id: {question_id}")
            if question_id not in path_length_map:
                raise KeyError(f"Missing path length for question_id: {question_id}")
            unpack_result(question_id, result)
        print(f"Found {len(results):,} existing scores, resuming...")

    for pred in tqdm(predictions, desc="LLM Match Evaluation"):
        question_id = pred["question_id"]

        if question_id in results:
            continue

        if question_id not in question_id_to_item:
            raise KeyError(f"question_id {question_id} not found in dataset")

        item = question_id_to_item[question_id]
        prediction = pred["answer"]
        extra_answers = item.get("extra_answers", None)

        score = evaluate_llm_match(
            question=item["question"],
            answer=item["answer"],
            prediction=prediction,
            extra_answers=extra_answers,
            model=config.model,
            dismiss_error=dismiss_error,
        )

        spl_score = compute_llm_match_spl_score(
            question_id,
            score,
            path_length_map=path_length_map,
            gt_path_length_map=gt_path_length_map,
        )
        results[question_id] = (score, spl_score)
        statistics = compute_statistics(
            results,
            total_questions=len(dataset),
            total_predictions=len(prediction_ids),
            blind_scores=blind_scores,
        )
        write_results(output_path, results, statistics)

    statistics = compute_statistics(
        results,
        total_questions=len(dataset),
        total_predictions=len(prediction_ids),
        blind_scores=blind_scores,
    )
    write_results(output_path, results, statistics)
    print(
        "Completed-only LLM Match: "
        f"{statistics['completed_only_llm_match_mean']:.2f}"
    )
    print(
        "Completed-only LLM Match SPL: "
        f"{statistics['completed_only_llm_match_spl_mean']:.2f}"
    )
    print(f"Original LLM Match: {statistics['original_llm_match_mean']:.2f}")
    print(
        "Original LLM Match SPL: "
        f"{statistics['original_llm_match_spl_mean']:.2f}"
    )
    print(f"Grounding LLM Match: {statistics['grounding_llm_match_mean']:.2f}")
    print(
        "Grounding LLM Match SPL: "
        f"{statistics['grounding_llm_match_spl_mean']:.2f}"
    )
    print(f"Prediction coverage: {statistics['prediction_coverage']:.2f}%")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path",
        default="cfg/openeqa.yaml", type=str,
    )
    parser.add_argument(
        "-d", "--dismiss-error", action="store_true",
        help="Set score to 1 if ValueError occurs",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-i", "--interactive", action="store_true",
        help="Interactively evaluate one question, answer, and prediction",
    )
    mode_group.add_argument(
        "--prepare", action="store_true",
        help="Prepare blind-LLM answers and judge scores for every dataset question",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    from omegaconf import OmegaConf

    load_dotenv(Path(__file__).resolve().parent.parent / "3dmem" / ".env")
    config = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(config)

    if args.interactive:
        interactive_main(config, dismiss_error=args.dismiss_error)
    elif args.prepare:
        prepare_main(config, dismiss_error=args.dismiss_error)
    else:
        config.output_dir = os.path.join(config.output_parent_dir, config.exp_name)
        main(config, dismiss_error=args.dismiss_error)
