import re
from pathlib import Path
import json
import argparse
import numpy as np


LABEL_DICT = {
    -1: 'Unknown',
    0: 'Excellent 5/5',
    1: 'Great 4.5/5',
    2: 'Good 4/5',
    3: 'Average 3.5/5',
    4: 'Bad 0-3/5',
}
TRAIN_LABELS_OCCURRENCES = {
    -1: -1,
    0: 3266, 1: 4539, 2: 2823, 3: 1080, 4: 538
}


def parse_result(result):
    if 'excellent' in result.lower():
        return LABEL_DICT[0]
    if 'great' in result.lower():
        return LABEL_DICT[1]
    if 'good' in result.lower():
        return LABEL_DICT[2]
    if 'average' in result.lower():
        return LABEL_DICT[3]
    if 'bad' in result.lower():
        return LABEL_DICT[4]
    # detect *.*/5 or */5 pattern
    pattern = r'([0-5](?:\.[0-5])?)/5'
    match = re.search(pattern, result)
    if match:
        rating = float(match.group(1))
        if rating >= 4.75:
            return LABEL_DICT[0]
        if rating >= 4.25:
            return LABEL_DICT[1]
        if rating >= 3.75:
            return LABEL_DICT[2]
        if rating >= 3.25:
            return LABEL_DICT[3]
        return LABEL_DICT[4]
    return "Unknown"


def main(args):
    root = Path(args.save_dir)
    n_data = 24492
    n_test_data = 6123
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

    # vote
    vote_idx = []
    vote_preds = []
    vote_targets = []
    vote_count = 0

    vote_ties = []
    vote_tie_possible_success = []
    vote_tie_breaking_success = []
    for test_data_idx, preds in all_results.items():
        results = [preds[key]['result'] for key in preds.keys()]
        targets = [preds[key]['target'] for key in preds.keys()]
        assert [targets[0]] * len(targets) == targets
        votes = np.array([parse_result(result['generation']['content'] if isinstance(result, dict) else result) for result in results])

        vote_result = np.unique(votes, return_counts=True)

        # break ties using train data occurrences
        if len(vote_result[0]) > 1 and np.sum(vote_result[1] == np.max(vote_result[1])) > 1:
            vote_ties.append(test_data_idx)
            tied_indices = np.where(vote_result[1] == vote_result[1].max())[0]
            tied_preds = [vote_result[0][tied_idx] for tied_idx in tied_indices]
            if targets[0] in tied_preds:
                vote_tie_possible_success.append(test_data_idx)
            max_key = None
            max_occurrence = -1
            for key, value in LABEL_DICT.items():
                if value in tied_preds and TRAIN_LABELS_OCCURRENCES[key] > max_occurrence:
                    max_key = key
                    max_occurrence = TRAIN_LABELS_OCCURRENCES[key]
            vote_result = LABEL_DICT[max_key]
            if vote_result == targets[0]:
                vote_tie_breaking_success.append(test_data_idx)
        else:
            vote_result = vote_result[0][np.argmax(vote_result[1])]

        vote_idx.append(test_data_idx)
        vote_preds.append(vote_result)
        vote_targets.append(targets[0])
        vote_count += len(results)

    accuracy = sum([p == t for p, t in zip(vote_preds, vote_targets)]) / len(vote_targets)

    print(f"Accuracy: {accuracy * 100:.2f}% / " \
          f"Progress: {vote_count / (n_test_data * n_preds) * 100:.2f}% ({len(vote_targets)}/{n_test_data})\n" \
          f"{root.as_posix()}\n" \
          f"Tie-breaking success: {len(vote_tie_breaking_success)} out of {len(vote_tie_possible_success)}/{len(vote_ties)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--n_preds', type=int, default=5)
    args_ = parser.parse_args()
    main(args_)
