//! Rust implementation of CTC forced alignment for word-level transcription.

use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// ─── Interpolation ──────────────────────────────────────────────────────────

fn interpolate_nans_nearest(vals: &mut [f64]) {
    let n = vals.len();
    if n == 0 {
        return;
    }

    let known: Vec<usize> = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, _)| i)
        .collect();

    if known.is_empty() {
        return;
    }
    if known.len() == 1 {
        let v = vals[known[0]];
        for x in vals.iter_mut() {
            if x.is_nan() {
                *x = v;
            }
        }
        return;
    }

    for i in 0..n {
        if !vals[i].is_nan() {
            continue;
        }
        let mut best_dist = usize::MAX;
        let mut best_val = f64::NAN;
        for &ki in &known {
            let dist = if i >= ki { i - ki } else { ki - i };
            if dist < best_dist || (dist == best_dist && ki < i) {
                best_dist = dist;
                best_val = vals[ki];
            }
        }
        vals[i] = best_val;
    }
}

// ─── Python-visible result types ────────────────────────────────────────────

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct WordSegment {
    #[pyo3(get)]
    word: String,
    #[pyo3(get)]
    start: f64, // NAN if missing
    #[pyo3(get)]
    end: f64,
    #[pyo3(get)]
    score: f64,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct CharSegment {
    #[pyo3(get)]
    char: String,
    #[pyo3(get)]
    start: f64,
    #[pyo3(get)]
    end: f64,
    #[pyo3(get)]
    score: f64,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct SubSegment {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    start: f64,
    #[pyo3(get)]
    end: f64,
    #[pyo3(get)]
    words: Vec<WordSegment>,
    #[pyo3(get)]
    chars: Option<Vec<CharSegment>>,
}

// ─── Core DP (pure Rust, no Python objects) ─────────────────────────────────

struct MergedSeg {
    start: i64,
    end: i64,
    score: f32,
}

fn dp_align(
    emission: &ndarray::ArrayView2<f32>,
    tokens: &[i32],
    blank_id: i32,
) -> Option<Vec<MergedSeg>> {
    let num_frames = emission.nrows();
    let num_tokens = tokens.len();

    let mut trellis = Array2::<f32>::zeros((num_frames + 1, num_tokens + 1));

    {
        let mut cum = 0.0f32;
        for t in 0..num_frames {
            cum += emission[(t, blank_id as usize)];
            trellis[(t + 1, 0)] = cum;
        }
    }
    for j in 1..=num_tokens {
        trellis[(0, j)] = f32::NEG_INFINITY;
    }
    if num_frames + 1 > num_tokens {
        let start = num_frames + 1 - num_tokens;
        for t in start..=num_frames {
            trellis[(t, 0)] = f32::INFINITY;
        }
    }

    for t in 0..num_frames {
        let blank_score = emission[(t, blank_id as usize)];
        for j in 0..num_tokens {
            let stay = trellis[(t, j + 1)] + blank_score;
            let change = trellis[(t, j)] + emission[(t, tokens[j] as usize)];
            trellis[(t + 1, j + 1)] = stay.max(change);
        }
    }

    let mut j = num_tokens;
    let mut t_start = 0usize;
    let mut best = f32::NEG_INFINITY;
    for t in 0..=num_frames {
        if trellis[(t, j)] > best {
            best = trellis[(t, j)];
            t_start = t;
        }
    }

    let mut path_token: Vec<usize> = Vec::with_capacity(t_start);
    let mut path_time: Vec<usize> = Vec::with_capacity(t_start);
    let mut path_score: Vec<f32> = Vec::with_capacity(t_start);

    let mut failed = true;
    for t in (1..=t_start).rev() {
        let stayed = trellis[(t - 1, j)] + emission[(t - 1, blank_id as usize)];
        let changed = trellis[(t - 1, j - 1)] + emission[(t - 1, tokens[j - 1] as usize)];

        let tok = if changed > stayed {
            tokens[j - 1]
        } else {
            blank_id
        };
        let prob = emission[(t - 1, tok as usize)].exp();

        path_token.push(j - 1);
        path_time.push(t - 1);
        path_score.push(prob);

        if changed > stayed {
            j -= 1;
            if j == 0 {
                failed = false;
                break;
            }
        }
    }

    if failed {
        return None;
    }

    path_token.reverse();
    path_time.reverse();
    path_score.reverse();

    let path_len = path_token.len();
    let mut segments: Vec<MergedSeg> = Vec::new();

    let mut i1 = 0usize;
    while i1 < path_len {
        let mut i2 = i1;
        while i2 < path_len && path_token[i1] == path_token[i2] {
            i2 += 1;
        }
        let count = (i2 - i1) as f32;
        let score: f32 = path_score[i1..i2].iter().sum::<f32>() / count;
        segments.push(MergedSeg {
            start: path_time[i1] as i64,
            end: path_time[i2 - 1] as i64 + 1,
            score,
        });
        i1 = i2;
    }

    Some(segments)
}

fn add_wildcard_col(emission: &ndarray::ArrayView2<f32>, blank_id: usize) -> (Array2<f32>, usize) {
    let (nframes, ncols) = emission.dim();
    let new_ncols = ncols + 1;
    let mut out = Array2::<f32>::zeros((nframes, new_ncols));

    for t in 0..nframes {
        let mut max_non_blank = f32::NEG_INFINITY;
        for c in 0..ncols {
            out[(t, c)] = emission[(t, c)];
            if c != blank_id && emission[(t, c)] > max_non_blank {
                max_non_blank = emission[(t, c)];
            }
        }
        out[(t, ncols)] = max_non_blank;
    }

    (out, ncols)
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn build_subsegments(
    char_segments: &[MergedSeg],
    text_chars: &[char],
    clean_cdx: &[usize],
    sentence_spans: &[(usize, usize)],
    ratio: f64,
    t1: f64,
    no_spaces: bool,
    return_char_alignments: bool,
) -> Vec<SubSegment> {
    let text_len = text_chars.len();

    let mut cdx_to_seg: Vec<Option<usize>> = vec![None; text_len];
    for (i, &cdx) in clean_cdx.iter().enumerate() {
        cdx_to_seg[cdx] = Some(i);
    }

    let mut char_starts: Vec<f64> = vec![f64::NAN; text_len];
    let mut char_ends: Vec<f64> = vec![f64::NAN; text_len];
    let mut char_scores: Vec<f64> = vec![f64::NAN; text_len];
    let mut char_word_idx: Vec<usize> = vec![0; text_len];

    let mut word_idx = 0usize;
    for cdx in 0..text_len {
        if let Some(seg_i) = cdx_to_seg[cdx] {
            let seg = &char_segments[seg_i];
            char_starts[cdx] = round3(seg.start as f64 * ratio + t1);
            char_ends[cdx] = round3(seg.end as f64 * ratio + t1);
            char_scores[cdx] = round3(seg.score as f64);
        }
        char_word_idx[cdx] = word_idx;
        if no_spaces {
            word_idx += 1;
        } else if cdx == text_len - 1 || text_chars[cdx + 1] == ' ' {
            word_idx += 1;
        }
    }

    let mut subsegments: Vec<SubSegment> = Vec::new();

    for &(sstart, send) in sentence_spans {
        let sentence_text: String = text_chars[sstart..send].iter().collect();

        let mut seen_word_idxs: Vec<usize> = Vec::new();
        for cdx in sstart..std::cmp::min(send + 1, text_len) {
            let wi = char_word_idx[cdx];
            if seen_word_idxs.is_empty() || *seen_word_idxs.last().unwrap() != wi {
                seen_word_idxs.push(wi);
            }
        }

        let mut sentence_start = f64::NAN;
        let mut sentence_end = f64::NAN;
        let mut sentence_words: Vec<WordSegment> = Vec::new();

        for &wi in &seen_word_idxs {
            let mut word_chars_text = String::new();
            let mut w_starts: Vec<f64> = Vec::new();
            let mut w_ends: Vec<f64> = Vec::new();
            let mut w_scores: Vec<f64> = Vec::new();

            for cdx in sstart..std::cmp::min(send + 1, text_len) {
                if char_word_idx[cdx] != wi {
                    continue;
                }
                word_chars_text.push(text_chars[cdx]);
                if text_chars[cdx] != ' ' {
                    if !char_starts[cdx].is_nan() {
                        w_starts.push(char_starts[cdx]);
                        if sentence_start.is_nan() || char_starts[cdx] < sentence_start {
                            sentence_start = char_starts[cdx];
                        }
                    }
                    if !char_ends[cdx].is_nan() {
                        w_ends.push(char_ends[cdx]);
                        if sentence_end.is_nan() || char_ends[cdx] > sentence_end {
                            sentence_end = char_ends[cdx];
                        }
                    }
                    if !char_scores[cdx].is_nan() {
                        w_scores.push(char_scores[cdx]);
                    }
                }
            }

            let word_text = word_chars_text.trim().to_string();
            if word_text.is_empty() {
                continue;
            }

            sentence_words.push(WordSegment {
                word: word_text,
                start: w_starts
                    .iter()
                    .copied()
                    .reduce(f64::min)
                    .unwrap_or(f64::NAN),
                end: w_ends.iter().copied().reduce(f64::max).unwrap_or(f64::NAN),
                score: if w_scores.is_empty() {
                    f64::NAN
                } else {
                    round3(w_scores.iter().sum::<f64>() / w_scores.len() as f64)
                },
            });
        }

        // Interpolate word timestamps
        if !sentence_words.is_empty() {
            let has_nan = sentence_words.iter().any(|w| w.start.is_nan());
            let has_val = sentence_words.iter().any(|w| !w.start.is_nan());
            if has_nan && has_val {
                let mut starts: Vec<f64> = sentence_words.iter().map(|w| w.start).collect();
                let mut ends: Vec<f64> = sentence_words.iter().map(|w| w.end).collect();
                interpolate_nans_nearest(&mut starts);
                interpolate_nans_nearest(&mut ends);
                for (i, w) in sentence_words.iter_mut().enumerate() {
                    if w.start.is_nan() && !starts[i].is_nan() {
                        w.start = starts[i];
                    }
                    if w.end.is_nan() && !ends[i].is_nan() {
                        w.end = ends[i];
                    }
                }
            }
        }

        let chars = if return_char_alignments {
            let mut chars_list = Vec::new();
            for cdx in sstart..std::cmp::min(send + 1, text_len) {
                chars_list.push(CharSegment {
                    char: text_chars[cdx].to_string(),
                    start: char_starts[cdx],
                    end: char_ends[cdx],
                    score: char_scores[cdx],
                });
            }
            Some(chars_list)
        } else {
            None
        };

        subsegments.push(SubSegment {
            text: sentence_text,
            start: sentence_start,
            end: sentence_end,
            words: sentence_words,
            chars,
        });
    }

    // Interpolate subsegment-level timestamps
    if !subsegments.is_empty() {
        let mut sub_starts: Vec<f64> = subsegments.iter().map(|s| s.start).collect();
        let mut sub_ends: Vec<f64> = subsegments.iter().map(|s| s.end).collect();
        interpolate_nans_nearest(&mut sub_starts);
        interpolate_nans_nearest(&mut sub_ends);
        for (i, s) in subsegments.iter_mut().enumerate() {
            s.start = sub_starts[i];
            s.end = sub_ends[i];
        }
    }

    // Merge adjacent subsegments with same (start, end) — they're time-ordered
    let joiner = if no_spaces { "" } else { " " };
    let mut merged: Vec<SubSegment> = Vec::new();
    for s in subsegments {
        if let Some(last) = merged.last_mut() {
            if last.start == s.start && last.end == s.end {
                last.text = format!("{}{}{}", last.text, joiner, s.text);
                last.words.extend(s.words);
                if let (Some(ref mut ec), Some(sc)) = (&mut last.chars, s.chars) {
                    ec.extend(sc);
                }
                continue;
            }
        }
        merged.push(s);
    }

    merged
}

// ─── Python-facing function ─────────────────────────────────────────────────

/// Full alignment pipeline for one segment. Releases the GIL for all heavy
/// computation (DP, timestamps, word grouping, interpolation).
///
/// Returns None if backtracking fails, otherwise a list of SubSegment objects.
#[pyfunction]
#[pyo3(signature = (
    emission,
    text,
    text_clean,
    model_dictionary,
    clean_cdx,
    sentence_spans,
    blank_id,
    t1,
    duration,
    no_spaces,
    return_char_alignments,
))]
fn align_segment<'py>(
    py: Python<'py>,
    emission: PyReadonlyArray2<'py, f32>,
    text: &str,
    text_clean: &str,
    model_dictionary: &Bound<'py, PyDict>,
    clean_cdx: Vec<usize>,
    sentence_spans: Vec<(usize, usize)>,
    blank_id: i32,
    t1: f64,
    duration: f64,
    no_spaces: bool,
    return_char_alignments: bool,
) -> PyResult<Option<Vec<SubSegment>>> {
    let emission_owned = emission.as_array().to_owned();
    let num_frames = emission_owned.nrows();

    let mut dict_map: HashMap<char, i32> = HashMap::new();
    for (key, value) in model_dictionary.iter() {
        let k: String = key.extract()?;
        let v: i32 = value.extract()?;
        let mut chars = k.chars();
        if let Some(ch) = chars.next() {
            if chars.next().is_none() {
                dict_map.insert(ch, v);
            }
        }
    }

    let text_chars: Vec<char> = text.chars().collect();
    let text_clean_chars: Vec<char> = text_clean.chars().collect();

    let result = py.detach(move || {
        let has_wildcard = text_clean_chars.iter().any(|c| !dict_map.contains_key(c));

        let (final_emission, tokens);
        if has_wildcard {
            let (new_em, wildcard_id) = add_wildcard_col(&emission_owned.view(), blank_id as usize);
            tokens = text_clean_chars
                .iter()
                .map(|c| *dict_map.get(c).unwrap_or(&(wildcard_id as i32)))
                .collect::<Vec<_>>();
            final_emission = new_em;
        } else {
            tokens = text_clean_chars
                .iter()
                .map(|c| dict_map[c])
                .collect::<Vec<_>>();
            final_emission = emission_owned;
        }

        let char_segments = match dp_align(&final_emission.view(), &tokens, blank_id) {
            Some(s) => s,
            None => return None,
        };

        let ratio = duration / (num_frames as f64);
        let subsegments = build_subsegments(
            &char_segments,
            &text_chars,
            &clean_cdx,
            &sentence_spans,
            ratio,
            t1,
            no_spaces,
            return_char_alignments,
        );

        Some(subsegments)
    });

    Ok(result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(align_segment, m)?)?;
    m.add_class::<SubSegment>()?;
    m.add_class::<WordSegment>()?;
    m.add_class::<CharSegment>()?;
    Ok(())
}

