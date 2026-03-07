"""Browse split_with_embeddings.jsonl: sample by id, pianorolls, 10 nearest by cosine, concatenations."""

import json

import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll


def parse_notes(
    notes_str: str,
) -> pd.DataFrame:
    try:
        data = json.loads(
            notes_str,
        )
        if not data:
            return pd.DataFrame()
        if isinstance(
            data,
            dict,
        ) and any(
            isinstance(
                v,
                list,
            )
            for v in data.values()
        ):
            return pd.DataFrame(
                data,
            )
        if isinstance(
            data,
            dict,
        ) and all(k in data for k in ["pitch", "start"]):
            return pd.DataFrame(
                data,
            )
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _parse_jsonl_lines(lines: list[str]) -> list[dict]:
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_embeddings_jsonl_from_upload(uploaded_file) -> list[dict]:
    """Load embeddings from an uploaded JSONL file (e.g. st.file_uploader)."""
    content = uploaded_file.getvalue().decode("utf-8")
    return _parse_jsonl_lines(content.splitlines())


def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    va = np.array(
        a,
        dtype=np.float64,
    )
    vb = np.array(
        b,
        dtype=np.float64,
    )
    na = np.linalg.norm(
        va,
    )
    nb = np.linalg.norm(
        vb,
    )
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(
        np.dot(
            va,
            vb,
        )
        / (na * nb),
    )


def render_concatenation(
    notes_first_str: str,
    notes_second_str: str,
    label: str,
) -> None:
    df_first = parse_notes(
        notes_first_str,
    )
    df_second = parse_notes(
        notes_second_str,
    )
    if df_first.empty and df_second.empty:
        st.caption(
            f"{label}: empty",
        )
        return
    if df_first.empty or df_second.empty:
        if not df_first.empty:
            piece = ff.MidiPiece(
                df_first,
            )
        else:
            piece = ff.MidiPiece(
                df_second,
            )
        st.caption(
            label,
        )
        streamlit_pianoroll.from_fortepyan(
            piece=piece,
        )
        return
    t_end_first = df_first["end"].max() if "end" in df_first.columns else df_first["start"].max()
    df_second_shifted = df_second.copy()
    if "start" in df_second_shifted.columns:
        df_second_shifted["start"] = df_second_shifted["start"] + t_end_first
    if "end" in df_second_shifted.columns:
        df_second_shifted["end"] = df_second_shifted["end"] + t_end_first

    piece_first = ff.MidiPiece(
        df_first,
    )
    piece_second = ff.MidiPiece(
        df_second_shifted,
    )
    st.caption(
        label,
    )
    streamlit_pianoroll.from_fortepyan(
        piece=piece_first,
        secondary_piece=piece_second,
    )


def main() -> None:
    st.set_page_config(
        page_title="Embeddings Browse",
        layout="wide",
    )
    st.title("Embeddings Browse")
    st.caption(
        "Pick sample by id: notes_first & notes_second pianorolls, their concatenation, "
        "then 10 most similar — each shows that sample's notes_first and "
        "chosen notes_first + that notes_first.",
    )

    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader(
        "Upload JSONL file",
        type=["jsonl", "json"],
        help="Upload split_with_embeddings.jsonl (from scripts.export_embeddings_jsonl).",
    )

    if uploaded is None:
        st.info("Upload a JSONL file (split_with_embeddings.jsonl) to browse embeddings.")
        return

    with st.spinner("Loading ..."):
        data = load_embeddings_jsonl_from_upload(uploaded)

    if not data:
        st.warning(
            "File is empty.",
        )
        return

    id_to_idx = {row["id"]: i for i, row in enumerate(data)}
    valid_ids = sorted(
        id_to_idx.keys(),
    )

    st.sidebar.write(
        f"Loaded **{len(data)}** samples. Ids: {valid_ids[0]} .. {valid_ids[-1]}",
    )
    sample_id = st.sidebar.number_input(
        "Sample id",
        min_value=valid_ids[0],
        max_value=valid_ids[-1],
        value=valid_ids[0],
        step=1,
    )

    if sample_id not in id_to_idx:
        st.error(
            f"Id {sample_id} not in dataset.",
        )
        return

    idx = id_to_idx[sample_id]
    row = data[idx]
    emb = row.get(
        "embedding",
        [],
    )
    if not emb:
        st.warning(
            f"Sample {sample_id} has no embedding (empty notes_first).",
        )

    st.header(
        f"Sample id = {sample_id}",
    )
    st.caption(
        f"source_dataset: {row.get('source_dataset', '—')} · original_id: {row.get('original_id', '—')}",
    )

    col_first, col_second = st.columns(2)
    with col_first:
        st.write("**notes_first**")
        df_first = parse_notes(row.get("notes_first", ""))
        if not df_first.empty:
            streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(df_first))
        else:
            st.caption("empty")
    with col_second:
        st.write("**notes_second**")
        df_second = parse_notes(row.get("notes_second", ""))
        if not df_second.empty:
            streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(df_second))
        else:
            st.caption("empty")

    st.subheader("Selected: concatenated (notes_first + notes_second)")
    render_concatenation(
        row.get("notes_first", ""),
        row.get("notes_second", ""),
        label=f"id={sample_id}",
    )

    nearest_10 = []
    if emb:
        sims = []
        for i, r in enumerate(
            data,
        ):
            e = r.get(
                "embedding",
                [],
            )
            if not e or i == idx:
                continue
            sim = cosine_similarity(
                emb,
                e,
            )
            sims.append(
                (
                    i,
                    r["id"],
                    sim,
                ),
            )
        sims.sort(
            key=lambda x: -x[2],
        )
        nearest_10 = sims[:10]
        least_10 = list(reversed(sims[-10:]))  # least similar first

    if emb:
        st.header("10 most similar (by embedding)")
        st.caption(
            "Each example: that sample's notes_first pianoroll, then concatenation of "
            "chosen notes_first + that notes_first."
        )
        if nearest_10:
            for rank, (i, rid, sim) in enumerate(nearest_10, 1):
                dist = 1.0 - sim
                st.sidebar.write(f"{rank}. id={rid} dist={dist:.4e}")
            chosen_notes_first = row.get("notes_first", "")
            for rank, (i, rid, sim) in enumerate(nearest_10, 1):
                r = data[i]
                other_notes_first = r.get("notes_first", "")
                dist = 1.0 - sim
                st.divider()
                st.subheader(f"Nearest #{rank} — id={rid} (dist={dist:.4e})")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**That sample's notes_first**")
                    df_other = parse_notes(other_notes_first)
                    if not df_other.empty:
                        streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(df_other))
                    else:
                        st.caption("empty")
                with col_b:
                    st.write("**Chosen notes_first + that notes_first**")
                    render_concatenation(
                        chosen_notes_first,
                        other_notes_first,
                        label="",
                    )
        else:
            st.caption("No other samples with embeddings.")

        st.header("10 least similar (by embedding)")
        if least_10:
            for rank, (i, rid, sim) in enumerate(least_10, 1):
                dist = 1.0 - sim
                st.sidebar.write(f"Least {rank}. id={rid} dist={dist:.4e}")
            chosen_notes_first = row.get("notes_first", "")
            for rank, (i, rid, sim) in enumerate(least_10, 1):
                r = data[i]
                other_notes_first = r.get("notes_first", "")
                dist = 1.0 - sim
                st.divider()
                st.subheader(f"Least #{rank} — id={rid} (dist={dist:.4e})")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**That sample's notes_first**")
                    df_other = parse_notes(other_notes_first)
                    if not df_other.empty:
                        streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(df_other))
                    else:
                        st.caption("empty")
                with col_b:
                    st.write("**Chosen notes_first + that notes_first**")
                    render_concatenation(
                        chosen_notes_first,
                        other_notes_first,
                        label="",
                    )
        else:
            st.caption("No other samples with embeddings.")
    else:
        st.info("No embedding for this sample; cannot compute nearest.")

    st.divider()
    with st.expander(
        "Embedding shape and truncated values",
    ):
        if emb:
            st.write(
                "**Shape:**",
                len(emb),
            )
            st.write(
                "**Truncated (first 50):**",
            )
            trunc = emb[:50]
            st.text(
                str(
                    trunc,
                ),
            )
        else:
            st.write(
                "No embedding.",
            )


if __name__ == "__main__":
    main()
