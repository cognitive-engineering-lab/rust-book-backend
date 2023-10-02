use std::{fmt, io::Read, sync::OnceLock};

use anyhow::{anyhow, Context, Result};
use arrayvec::ArrayString;
use flate2::read::ZlibDecoder;

use fxhash::FxHashSet as HashSet;
use indexical::define_index_type;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;

use pyo3::{
  prelude::*,
  types::{PyDict, PyList},
};
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{
  quizzes::{self, MultipleChoice, Question, Quizzes, Schema},
  time,
};

type DateTime = chrono::DateTime<chrono::Utc>;
pub type CommitHash = ArrayString<41>;
pub type QuizName = ArrayString<60>;
type QuizHash = ArrayString<32>;

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Hash)]
pub struct QuestionId(ArrayString<36>);

impl QuestionId {
  pub fn new(s: &str) -> Result<Self> {
    Ok(QuestionId(
      ArrayString::from(s).map_err(|_| anyhow!("Question id too long"))?,
    ))
  }
}

impl fmt::Debug for QuestionId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.0.fmt(f)
  }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Hash)]
pub struct SessionId(ArrayString<40>);

impl fmt::Debug for SessionId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.0.fmt(f)
  }
}

define_index_type! {
  pub struct QuestionIndex for QuestionId = u16;
}

define_index_type! {
  pub struct SessionIndex for SessionId = u32;
}

#[derive(Deserialize, PartialOrd, Ord, PartialEq, Eq, Clone, Copy)]
struct UtcMillis(i64);

impl TryFrom<UtcMillis> for DateTime {
  type Error = anyhow::Error;
  fn try_from(value: UtcMillis) -> Result<Self, Self::Error> {
    chrono::DateTime::from_timestamp(value.0 / 1000, ((value.0 % 1000) * 1_000_000) as u32)
      .ok_or(anyhow!("Timestamp conversion failed"))
  }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TelemetryRaw {
  session_id: SessionId,
  commit_hash: CommitHash,
  timestamp: UtcMillis,
  payload: QuizResponseRaw,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct QuizResponseRaw {
  quiz_name: QuizName,
  quiz_hash: QuizHash,
  answers: SmallVec<[Option<QuestionResponseRaw>; 4]>,
  attempt: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct QuestionResponseRaw {
  answer: AnswerContent,
  correct: bool,
  start: Option<UtcMillis>,
  end: Option<UtcMillis>,
  explanation: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", untagged)]
pub enum AnswerContent {
  Tracing(quizzes::TracingAnswer),
  ShortAnswer(quizzes::ShortAnswerAnswer),
  MultipleChoice(quizzes::MultipleChoiceAnswer),
  MultipleChoiceOld(quizzes::MultipleChoiceOldAnswer),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Answer {
  pub session_id: SessionId,
  pub commit_hash: CommitHash,
  pub timestamp: DateTime,
  pub quiz_name: QuizName,
  pub chapter: usize,
  pub section: usize,
  pub question: u16,
  pub question_id: Option<QuestionId>,
  pub answer: AnswerContent,
  pub correct: u16,
  pub start: Option<DateTime>,
  pub end: Option<DateTime>,
  pub duration: Option<usize>,
  pub version: usize,
  pub explanation: Option<String>,
}

fn load_raw_data() -> Result<Vec<TelemetryRaw>> {
  let conn = sqlite::open("../data/log.sqlite")?;
  let stmt = conn.prepare("SELECT data FROM answers")?;
  let raw_data = stmt
    .into_iter()
    .map(|res| {
      let sql_row = res?;
      let data = sql_row.read::<&[u8], _>("data");
      Ok(data.to_vec())
    })
    .collect::<sqlite::Result<Vec<_>>>()?;

  raw_data
    .into_par_iter()
    .progress_with_style(crate::pb_style())
    .map(|data| {
      let z = ZlibDecoder::new(data.as_slice());
      let raw: TelemetryRaw = serde_json::from_reader(z).with_context(|| {
        let mut s = String::new();
        ZlibDecoder::new(data.as_slice())
          .read_to_string(&mut s)
          .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        serde_json::to_string_pretty(&v).unwrap()
      })?;
      Ok(raw)
    })
    .collect()
}

pub fn load_answers_rs() -> Result<(Quizzes, Vec<Answer>)> {
  let mut raw_data = time!("raw", load_raw_data()?);

  // Remove example data
  raw_data.retain(|row| row.payload.quiz_name.as_str() != "example-quiz");

  // Only keep the first attempt
  raw_data.retain(|row| row.payload.attempt.unwrap_or(0) == 0);

  let input = raw_data
    .iter()
    .map(|row| (&row.payload.quiz_name, &row.commit_hash))
    .collect::<Vec<_>>();
  let (quizzes, ignored) = time!("quizzes", Quizzes::build(input)?);

  // Remove inputs with stale commit hashes
  let mut i = 0;
  raw_data.retain(|_| {
    let keep = !ignored.contains(&i);
    i += 1;
    keep
  });

  // Only keep the first complete answer for a given user/quiz pair
  let first_attempts = time!("first_attempts", {
    let groups = raw_data
      .iter()
      .enumerate()
      .filter(|(_, row)| {
        let quiz = quizzes.schema_ref(&row.payload.quiz_name, &row.commit_hash);
        quiz.questions.len() == row.payload.answers.len()
      })
      .map(|(i, row)| {
        (
          (row.session_id, row.payload.quiz_name, row.payload.quiz_hash),
          (i, row.timestamp),
        )
      })
      .into_group_map();

    groups
      .into_values()
      .map(|pairs: Vec<(usize, UtcMillis)>| pairs.into_iter().min_by_key(|(_, ts)| *ts).unwrap().0)
      .collect::<HashSet<_>>()
  });

  let mut i = 0;
  raw_data.retain(|_| {
    let keep = first_attempts.contains(&i);
    i += 1;
    keep
  });

  let versions = raw_data
    .iter()
    .map(|row| quizzes.version(&row.payload.quiz_name, &row.commit_hash))
    .collect::<Result<Vec<_>>>()?;

  let data = raw_data
    .into_iter()
    .zip(versions)
    .flat_map(|(row, version)| {
      let schema = quizzes.schema_ref(&row.payload.quiz_name, &row.commit_hash);
      let answers = row.payload.answers.into_iter().enumerate();
      answers
        .filter_map(move |(i, raw_ans)| {
          let raw_ans = raw_ans?;
          let correct = patch_tracing_answers(&raw_ans, schema, i)?;
          Some((i, raw_ans, correct))
        })
        .map(move |(i, raw_ans, correct)| {
          let start: Option<DateTime> = match raw_ans.start {
            Some(t) => Some(t.try_into()?),
            None => None,
          };
          let end: Option<DateTime> = match raw_ans.end {
            Some(t) => Some(t.try_into()?),
            None => None,
          };
          let duration = match (start.as_ref(), end.as_ref()) {
            (Some(start), Some(end)) => {
              Some(end.signed_duration_since(start).num_seconds() as usize)
            }
            _ => None,
          };

          static REGEX: OnceLock<Regex> = OnceLock::new();
          let regex = REGEX.get_or_init(|| Regex::new("^ch([0-9]+)-([0-9]+)").unwrap());

          let captures = regex
            .captures(&row.payload.quiz_name)
            .with_context(|| format!("Failed to match quiz name: {}", row.payload.quiz_name))?;
          let chapter_str = captures.get(1).unwrap().as_str();
          let chapter = chapter_str
            .parse::<usize>()
            .with_context(|| format!("Failed to parse: {chapter_str}"))?;
          let section_str = captures.get(2).unwrap().as_str();
          let section = section_str
            .parse::<usize>()
            .with_context(|| format!("Failed to parse: {section_str}"))?;

          Ok(Answer {
            session_id: row.session_id,
            commit_hash: row.commit_hash,
            timestamp: row.timestamp.try_into()?,
            quiz_name: row.payload.quiz_name,
            chapter,
            section,
            question: i as u16,
            question_id: match &schema.questions[i] {
              Question::Tracing(q) => q.id,
              Question::ShortAnswer(q) => q.id,
              Question::MultipleChoice(MultipleChoice::MultipleChoiceNew(q)) => q.id,
              Question::MultipleChoice(MultipleChoice::MultipleChoiceOld(q)) => q.id,
            },
            start,
            end,
            duration,
            version,
            correct: if correct { 1 } else { 0 },
            answer: raw_ans.answer,
            explanation: raw_ans.explanation,
          })
        })
    })
    .collect::<Result<Vec<_>>>()?;

  Ok((quizzes, data))
}

fn patch_tracing_answers(raw_ans: &QuestionResponseRaw, schema: &Schema, i: usize) -> Option<bool> {
  Some(match &raw_ans.answer {
    AnswerContent::Tracing(tracing_ans) => {
      let Question::Tracing(tracing_q) = &schema.questions[i] else {
        return None
      };
      if !tracing_q.answer.does_compile && !tracing_ans.does_compile {
        true
      } else {
        raw_ans.correct
      }
    }
    _ => raw_ans.correct,
  })
}

#[pyfunction]
pub fn load_answers(py: Python) -> PyResult<(Quizzes, PyObject)> {
  let (quizzes, answers) = load_answers_rs()?;

  let answers_py = PyList::new(
    py,
    answers
      .into_iter()
      .map(|ans| {
        let obj = pythonize::pythonize(py, &ans)?;
        let dict = obj.downcast::<PyDict>(py).unwrap();
        if let Some(start) = ans.start {
          dict.set_item("start", start.to_object(py))?;
        }
        if let Some(end) = ans.end {
          dict.set_item("end", end.to_object(py))?;
        }
        dict.set_item("timestamp", ans.timestamp.to_object(py))?;
        Ok(obj)
      })
      .collect::<Result<Vec<_>>>()?,
  )
  .to_object(py);

  Ok((quizzes, answers_py))
}
