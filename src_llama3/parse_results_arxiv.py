import re
from pathlib import Path
import json
import argparse
import numpy as np


LABEL_DICT = {
    -1: 'Unknown',
    0: 'Numerical Analysis (cs.NA)',
    1: 'Multimedia (cs.MM)',
    2: 'Logic in Computer Science (cs.LO)',
    3: 'Computers and Society (cs.CY)',
    4: 'Cryptography and Security (cs.CR)',
    5: 'Distributed, Parallel, and Cluster Computing (cs.DC)',
    6: 'Human-Computer Interaction (cs.HC)',
    7: 'Computational Engineering, Finance, and Science (cs.CE)',
    8: 'Networking and Internet Architecture (cs.NI)',
    9: 'Computational Complexity (cs.CC)',
    10: 'Artificial Intelligence (cs.AI)',
    11: 'Multiagent Systems (cs.MA)',
    12: 'General Literature (cs.GL)',
    13: 'Neural and Evolutionary Computing (cs.NE)',
    14: 'Symbolic Computation (cs.SC)',
    15: 'Hardware Architecture (cs.AR)',
    16: 'Computer Vision and Pattern Recognition (cs.CV)',
    17: 'Graphics (cs.GR)',
    18: 'Emerging Technologies (cs.ET)',
    19: 'Systems and Control (cs.SY)',
    20: 'Computational Geometry (cs.CG)',
    21: 'Other Computer Science (cs.OH)',
    22: 'Programming Languages (cs.PL)',
    23: 'Software Engineering (cs.SE)',
    24: 'Machine Learning (cs.LG)',
    25: 'Sound (cs.SD)',
    26: 'Social and Information Networks (cs.SI)',
    27: 'Robotics (cs.RO)',
    28: 'Information Theory (cs.IT)',
    29: 'Performance (cs.PF)',
    30: 'Computation and Language (cs.CL)',
    31: 'Information Retrieval (cs.IR)',
    32: 'Mathematical Software (cs.MS)',
    33: 'Formal Languages and Automata Theory (cs.FL)',
    34: 'Data Structures and Algorithms (cs.DS)',
    35: 'Operating Systems (cs.OS)',
    36: 'Computer Science and Game Theory (cs.GT)',
    37: 'Databases (cs.DB)',
    38: 'Digital Libraries (cs.DL)',
    39: 'Discrete Mathematics (cs.DM)',
}
TRAIN_LABELS_OCCURRENCES = {
    -1: -1, # -1: 78402,
    0: 437, 1: 382, 2: 3604, 3: 1014, 4: 2864, 5: 2933, 6: 703, 7: 380, 8: 4056, 9: 2245, 10: 5182,
    11: 391, 12: 21, 13: 1290, 14: 473, 15: 248, 16: 9998, 17: 202, 18: 402, 19: 1873, 20: 1495,
    21: 304, 22: 1268, 23: 1539, 24: 6989, 25: 457, 26: 2854, 27: 1661, 28: 16284, 29: 239, 30: 4334,
    31: 1350, 32: 270, 33: 926, 34: 5426, 35: 75, 36: 2506, 37: 1615, 38: 1100, 39: 1551
}


def parse_result(result):
    start = result.find("cs.")
    if start == -1:
        return "Unknown"
    area_code = result[start:start + 5]
    for value in LABEL_DICT.values():
        if area_code.lower() in value.lower():
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
    n_data = 169343
    n_test_data = 48603
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
        if args.on_subset:
            if test_data_idx > 775:
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
        if 'zero_shot' in args.save_dir:
            all_results[test_data_idx][pred_idx]['num_papers'] = 0
        elif 'one_shot' in args.save_dir:
            all_results[test_data_idx][pred_idx]['num_papers'] = 40
        elif isinstance(json_data['dialog'], str):
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
          f"Num Papers: {np.mean(num_papers):.2f} / {np.median(num_papers):.2f} / {np.std(num_papers):.2f}\n" \
          f"{root.as_posix()}\n" \
          f"Tie-breaking success: {len(vote_tie_breaking_success)} out of {len(vote_tie_possible_success)}/{len(vote_ties)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--n_preds', type=int, default=5)
    parser.add_argument('--on_subset', action='store_true')
    args_ = parser.parse_args()
    main(args_)
