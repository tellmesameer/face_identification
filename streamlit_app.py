import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_FILE = DATA_DIR / "results" / "matched_paths.json"
FEEDBACK_FILE = DATA_DIR / "results" / "match_feedback.json"


def load_json(path: Path, default):
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_matches():
    data = load_json(RESULTS_FILE, {"match_count": 0, "matches": []})
    matches = data.get("matches", [])

    normalized_matches = []
    for match in matches:
        image_path = match.get("image_path")
        if not image_path:
            continue

        normalized_matches.append(
            {
                "image_path": image_path,
                "confidence": match.get("confidence"),
                "centroid_distance": match.get("centroid_distance"),
                "min_reference_distance": match.get("min_reference_distance"),
                "mean_reference_distance": match.get("mean_reference_distance"),
            }
        )

    return normalized_matches


@st.cache_data(show_spinner=False)
def load_feedback():
    data = load_json(FEEDBACK_FILE, {"labels": {}})
    return data.get("labels", {})


def save_feedback(labels, matches):
    positive = []
    negative = []

    for match in matches:
        image_path = match["image_path"]
        vote = labels.get(image_path, {}).get("vote", "unreviewed")

        item = {
            "image_path": image_path,
            "vote": vote,
            "confidence": match.get("confidence"),
            "centroid_distance": match.get("centroid_distance"),
            "min_reference_distance": match.get("min_reference_distance"),
            "mean_reference_distance": match.get("mean_reference_distance"),
            "reviewed_at": labels.get(image_path, {}).get("reviewed_at"),
        }

        if vote == "correct":
            positive.append(item)
        elif vote == "incorrect":
            negative.append(item)

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_matches": len(matches),
            "reviewed": len(positive) + len(negative),
            "correct": len(positive),
            "incorrect": len(negative),
            "unreviewed": len(matches) - len(positive) - len(negative),
        },
        "labels": labels,
        "training_candidates": {
            "positive_matches": positive,
            "negative_matches": negative,
        },
    }
    save_json(FEEDBACK_FILE, payload)


def ensure_session_state(matches):
    match_paths = tuple(match["image_path"] for match in matches)
    current_signature = (RESULTS_FILE.as_posix(), match_paths)

    if st.session_state.get("match_signature") != current_signature:
        st.session_state.match_signature = current_signature
        st.session_state.labels = dict(load_feedback())
        st.session_state.feedback_dirty = False
        st.session_state.last_feedback_action = None


def queue_vote(image_path: str, vote: str):
    reviewed_at = datetime.now(timezone.utc).isoformat()
    st.session_state.labels[image_path] = {
        "vote": vote,
        "reviewed_at": reviewed_at,
    }
    st.session_state.feedback_dirty = True
    st.session_state.last_feedback_action = f"Marked {Path(image_path).name} as {vote}."


def clear_vote(image_path: str):
    st.session_state.labels.pop(image_path, None)
    st.session_state.feedback_dirty = True
    st.session_state.last_feedback_action = f"Cleared review for {Path(image_path).name}."


def flush_feedback(matches):
    if not st.session_state.get("feedback_dirty"):
        return

    save_feedback(st.session_state.labels, matches)
    load_feedback.clear()
    st.session_state.feedback_dirty = False


def main():
    st.set_page_config(page_title="Face Match Review", layout="wide")
    st.title("Face Match Review")
    st.caption("Review matched images and save labels for future threshold tuning or retraining workflows.")

    matches = load_matches()
    labels = load_feedback()

    if not RESULTS_FILE.exists():
        st.error(f"Match file not found: {RESULTS_FILE.as_posix()}")
        st.stop()

    if not matches:
        st.info("No matches found in matched_paths.json. Run the matcher first.")
        st.stop()

    ensure_session_state(matches)
    flush_feedback(matches)

    labels = st.session_state.labels

    reviewed_count = sum(
        1 for match in matches if labels.get(match["image_path"], {}).get("vote") in {"correct", "incorrect"}
    )
    correct_count = sum(
        1 for match in matches if labels.get(match["image_path"], {}).get("vote") == "correct"
    )
    incorrect_count = sum(
        1 for match in matches if labels.get(match["image_path"], {}).get("vote") == "incorrect"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total matches", len(matches))
    col2.metric("Reviewed", reviewed_count)
    col3.metric("Correct", correct_count)
    col4.metric("Incorrect", incorrect_count)

    st.divider()

    unreviewed_matches = [
        match
        for match in matches
        if labels.get(match["image_path"], {}).get("vote") not in {"correct", "incorrect"}
    ]

    show_reviewed = st.checkbox("Show already marked results", value=False)
    visible_matches = matches if show_reviewed else unreviewed_matches

    if st.session_state.get("last_feedback_action"):
        st.toast(st.session_state.last_feedback_action)
        st.session_state.last_feedback_action = None

    if not show_reviewed:
        st.caption(f"Showing {len(unreviewed_matches)} unreviewed matches.")

    if not visible_matches:
        if unreviewed_matches:
            st.info("No matches available for the current filter.")
        else:
            st.success("All current matches are already marked.")
        st.caption(f"Feedback is saved to {FEEDBACK_FILE.as_posix()}")
        st.stop()

    for index, match in enumerate(visible_matches, start=1):
        image_path = match["image_path"]
        vote = labels.get(image_path, {}).get("vote", "unreviewed")
        image_file = Path(image_path)

        left, right = st.columns([1.2, 1])

        with left:
            st.subheader(f"{index}. {image_file.name}")
            if image_file.exists():
                st.image(str(image_file), width="stretch")
            else:
                st.warning("Image file not found on disk.")
                st.code(image_path)

        with right:
            st.write(f"Path: `{image_path}`")
            if match.get("confidence") is not None:
                st.write(f"Confidence: `{match['confidence']}`")
            if match.get("min_reference_distance") is not None:
                st.write(f"Min reference distance: `{match['min_reference_distance']}`")
            if match.get("mean_reference_distance") is not None:
                st.write(f"Mean reference distance: `{match['mean_reference_distance']}`")
            if match.get("centroid_distance") is not None:
                st.write(f"Centroid distance: `{match['centroid_distance']}`")
            st.write(f"Current vote: `{vote}`")

            button_cols = st.columns(3)
            button_cols[0].button(
                "Correct",
                key=f"correct_{index}",
                width="stretch",
                on_click=queue_vote,
                args=(image_path, "correct"),
            )
            button_cols[1].button(
                "Incorrect",
                key=f"incorrect_{index}",
                width="stretch",
                on_click=queue_vote,
                args=(image_path, "incorrect"),
            )
            button_cols[2].button(
                "Clear",
                key=f"clear_{index}",
                width="stretch",
                on_click=clear_vote,
                args=(image_path,),
            )

        st.divider()

    st.success(f"Feedback is saved to {FEEDBACK_FILE.as_posix()}")


if __name__ == "__main__":
    main()
