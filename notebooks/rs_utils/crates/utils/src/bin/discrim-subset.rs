use anyhow::Result;
use float_ord::FloatOrd;
use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};
use indexical::{
  impls::BitvecRefIndexSet as IndexSet, index_vec::IndexVec, IndexSetIteratorExt, IndexedDomain,
};
use indicatif::{ParallelProgressIterator, ProgressBar};
use itertools::Itertools;
use rayon::prelude::*;
use rgsl::statistics::correlation;
use serde::Serialize;
use std::env;
use utils::answers::{QuestionIndex, SessionId, SessionIndex};

fn n_choose_k(n: usize, k: usize) -> usize {
  ((n - k + 1)..=n).product::<usize>() / (1..=k).product::<usize>()
}

const THRESHOLD: usize = 50;

#[derive(Serialize)]
struct Correlation<Id> {
  qs: Vec<Id>,
  r: f64,
  n: usize,
}

fn mean(v: &[f64]) -> f64 {
  rgsl::statistics::mean(v, 1, v.len())
}

fn main() -> Result<()> {
  env_logger::init();

  let n_comb = env::args().nth(1).unwrap().parse::<usize>()?;
  let (_quizzes, mut answers_flat) = utils::answers::load_answers_rs()?;
  answers_flat.retain(|ans| ans.question_id.is_some());

  let all_qs = answers_flat
    .iter()
    .filter_map(|ans| ans.question_id)
    .collect::<HashSet<_>>();
  let q_domain = &IndexedDomain::from_iter(all_qs.iter().copied());

  let mut answers_by_session = answers_flat
    .iter()
    .map(|ans| (ans.session_id, ans))
    .into_group_map();

  let mean_ans = mean(
    &answers_by_session
      .values()
      .map(|v| v.len() as f64)
      .collect_vec(),
  );

  answers_by_session.retain(|_, v| v.len() as f64 >= mean_ans);

  let sess_domain = &IndexedDomain::from_iter(answers_by_session.keys().copied());
  let mean_scores =
    IndexVec::<SessionIndex, _>::from_iter(sess_domain.as_vec().iter().map(|sess| {
      mean(
        &answers_by_session[sess]
          .iter()
          .map(|ans| ans.correct as f64)
          .collect_vec(),
      )
    }));

  let score_matrix = IndexVec::<QuestionIndex, _>::from_iter(q_domain.as_vec().iter().map(|q| {
    let q_score_map = answers_flat
      .iter()
      .filter(|ans| ans.question_id == Some(*q) && sess_domain.contains(&ans.session_id))
      .map(|ans| (sess_domain.index(&ans.session_id), ans.correct))
      .collect::<HashMap<_, _>>();
    let q_score_vec = IndexVec::<SessionIndex, _>::from_iter(
      sess_domain
        .as_vec()
        .indices()
        .map(|sess| q_score_map.get(&sess).copied().unwrap_or(0)),
    );

    let sess_set: IndexSet<'_, SessionId> =
      q_score_map.keys().copied().collect_indices(&sess_domain);
    (q_score_vec, sess_set)
  }));

  let mut v = Vec::new();
  for (_, sess_set) in &score_matrix {
    v.push((sess_set.len() as f64) / (sess_domain.len() as f64));
  }
  println!("Mean sparsity: {:.3}", mean(&v));

  let inputs = q_domain
    .as_vec()
    .indices()
    .filter(|idx| score_matrix[*idx].1.len() > THRESHOLD)
    .collect::<Vec<_>>();
  let pb = ProgressBar::new(n_choose_k(inputs.len(), n_comb) as u64).with_style(utils::pb_style());

  let corrs = inputs
    .into_iter()
    .combinations(n_comb)
    .par_bridge()
    .progress_with(pb)
    .map_init(
      || {
        (
          Vec::with_capacity(sess_domain.len()),
          Vec::with_capacity(sess_domain.len()),
          IndexSet::new(&sess_domain),
        )
      },
      |(x, y, sess_set), qs| {
        sess_set.clone_from(&score_matrix[qs[0]].1);
        for q in &qs[1..] {
          let other_sess_set = &score_matrix[*q].1;
          sess_set.intersect(other_sess_set);
        }

        let n_sess = sess_set.len();
        if n_sess < THRESHOLD {
          return None;
        }

        x.clear();
        y.clear();
        for sess in sess_set.indices() {
          let mut n: usize = 0;
          for q in &qs {
            n += score_matrix[*q].0[sess] as usize;
          }
          let mean = (n as f64) / (n_comb as f64);

          x.push(mean);
          y.push(mean_scores[sess]);
        }

        let r = correlation(&*x, 1, &*y, 1, n_sess);
        if r.is_nan() {
          return None;
        }

        Some(Correlation { qs, r, n: n_sess })
      },
    )
    .filter_map(|x| {
      let x = x?;
      let mut v = Vec::with_capacity(10 * 2);
      v.push(x);
      Some(v)
    })
    .reduce(Vec::new, |mut v1, v2| {
      v1.extend(v2);
      v1.sort_by_key(|q| FloatOrd(-q.r));
      v1.truncate(10);
      v1
    });

  let corrs = corrs
    .into_iter()
    .map(|q| Correlation {
      qs: q.qs.iter().map(|q| q_domain.value(*q)).collect::<Vec<_>>(),
      r: q.r,
      n: q.n,
    })
    .collect_vec();

  println!("{}", serde_json::to_string_pretty(&corrs)?);

  Ok(())
}
