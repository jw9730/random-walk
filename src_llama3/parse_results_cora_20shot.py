import re
from pathlib import Path
import json
import argparse
import numpy as np


LABEL_DICT = {
    -1: 'Unknown',
    0: 'Theory',
    1: 'Reinforcement Learning',
    2: 'Genetic Algorithms',
    3: 'Neural Networks',
    4: 'Probabilistic Methods',
    5: 'Case-Based',
    6: 'Rule Learning',
}
TRAIN_LABELS_OCCURRENCES = {
    -1: -1,
    0: 20, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20
}


def parse_result(result):
    for _, value in LABEL_DICT.items():
        if value.lower() in result.lower() and result.lower().find(value.lower()) == 0:
            return value
    for _, value in LABEL_DICT.items():
        if value.lower() in result.lower():
            return value
    return "Unknown"


def get_num_papers(text):
    patterns = [
        r"Paper\s+(\d{1,2})\b",                    # Paper X
        r"Paper\s+(\d{1,2})\s+cites\s+(\d{1,2})",  # Paper X cites Y
        r"Paper\s+(\d{1,2})\s+is cited by\s+(\d{1,2})"  # Paper X is cited by Y
    ]

    paper_ids = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            if len(matches[0]) == 1:
                # For "Paper X" pattern
                for m in matches:
                    paper_ids.add(int(m[0]))
            else:
                # For citation patterns
                for m in matches:
                    paper_ids.add(int(m[0]))
                    paper_ids.add(int(m[1]))

    assert len(paper_ids) > 0 and list(paper_ids) == list(range(1, max(paper_ids) + 1))
    return max(paper_ids)


def main(args):
    root = Path(args.save_dir)
    n_data = 2708
    n_test_data = 2068
    n_preds = args.n_preds

    # read all .json files in root
    all_results = {}
    for file in root.glob("*.json"):
        test_data_idx, pred_idx = file.stem.split("_")

        test_data_idx = int(test_data_idx)
        pred_idx = int(pred_idx)

        assert 0 <= test_data_idx < n_data
        if pred_idx >= n_preds:
            continue

        if test_data_idx not in all_results:
            all_results[test_data_idx] = {}

        if pred_idx not in all_results[test_data_idx]:
            all_results[test_data_idx][pred_idx] = {}

        with open(file, "r", encoding='utf-8') as f:
            json_data = json.load(f)

        all_results[test_data_idx][pred_idx]['dialog'] = json_data['dialog']
        all_results[test_data_idx][pred_idx]['result'] = json_data['result']
        all_results[test_data_idx][pred_idx]['target'] = json_data['target']
        if isinstance(json_data['dialog'], str):
            all_results[test_data_idx][pred_idx]['num_papers'] = get_num_papers(json_data['dialog'])
        else:
            all_results[test_data_idx][pred_idx]['num_papers'] = get_num_papers(json_data['dialog'][1]['content'])

    # num_papers
    num_papers = []
    for test_data_idx, preds in all_results.items():
        for pred_idx, pred in preds.items():
            num_papers.append(pred['num_papers'])

    # vote
    vote_idx = []
    vote_preds = []
    vote_targets = []
    vote_count = 0

    for test_data_idx, preds in all_results.items():
        results = [preds[key]['result'] for key in preds.keys()]
        targets = [preds[key]['target'] for key in preds.keys()]
        assert [targets[0]] * len(targets) == targets
        votes = np.array([parse_result(result['generation']['content'] if isinstance(result, dict) else result) for result in results])

        vote_result = np.unique(votes, return_counts=True)
        # randomly break ties
        if len(vote_result[0]) > 1 and np.sum(vote_result[1] == np.max(vote_result[1])) > 1:
            np.random.seed(test_data_idx)
            vote_result = vote_result[0][np.random.choice(np.where(vote_result[1] == vote_result[1].max())[0])]
        else:
            vote_result = vote_result[0][np.argmax(vote_result[1])]

        vote_idx.append(test_data_idx)
        vote_preds.append(vote_result)
        vote_targets.append(targets[0])
        vote_count += len(results)

    accuracy = sum([p == t for p, t in zip(vote_preds, vote_targets)]) / len(vote_targets)

    print(f"Accuracy: {accuracy * 100:.2f}% / " \
          f"Progress: {vote_count / (n_test_data * n_preds) * 100:.2f}% ({len(vote_targets)}/{n_test_data})\n" \
          f"Num Papers: {np.mean(num_papers):.2f} / {np.median(num_papers):.2f} / {np.std(num_papers):.2f}\n" \
          f"{root.as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--n_preds', type=int, default=5)
    args_ = parser.parse_args()
    main(args_)
