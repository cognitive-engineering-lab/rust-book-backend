use crate::answers::{CommitHash, QuestionId, QuizName};
use crate::{time, token};

use anyhow::{anyhow, Context, Result};
use arrayvec::ArrayString;
use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};
use indicatif::ParallelProgressIterator;
use internment::Intern;
use itertools::Itertools;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

type Commits = HashMap<QuizName, HashMap<CommitHash, (u64, usize)>>;
#[pyclass]
pub struct Quizzes {
  schemas: HashMap<u64, Intern<Schema>>,
  commits: Commits,
  lengths: HashMap<QuestionId, usize>,
}

const RUST_BOOK_DIR: &str = "../code/rust-book";

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct Schema {
  pub questions: Vec<Question>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
enum QuestionType {
  Tracing,
  MultipleChoice,
  ShortAnswer,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct QuestionFields<Prompt, Answer> {
  pub id: Option<QuestionId>,
  pub prompt: Prompt,
  pub answer: Answer,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum MultipleChoice {
  MultipleChoiceNew(MultipleChoiceNew),
  MultipleChoiceOld(MultipleChoiceOld),
}

pub type MultipleChoiceNew = QuestionFields<MultipleChoicePrompt, MultipleChoiceAnswer>;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct MultipleChoicePrompt {
  pub prompt: String,
  pub distractors: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct MultipleChoiceAnswer {
  pub answer: MultipleChoiceAnswerOptions,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum MultipleChoiceAnswerOptions {
  Single(String),
  Multiple(Vec<String>),
}

pub type MultipleChoiceOld = QuestionFields<MultipleChoiceOldPrompt, MultipleChoiceOldAnswer>;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct MultipleChoiceOldPrompt {
  pub prompt: String,
  pub choices: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct MultipleChoiceOldAnswer {
  pub answer: usize,
}

pub type ShortAnswer = QuestionFields<ShortAnswerPrompt, ShortAnswerAnswer>;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct ShortAnswerPrompt {
  pub prompt: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct ShortAnswerAnswer {
  pub answer: String,
}

pub type Tracing = QuestionFields<TracingPrompt, TracingAnswer>;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct TracingPrompt {
  pub program: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct TracingAnswer {
  pub does_compile: bool,
  pub line_number: Option<usize>,
  pub stdout: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(tag = "type")]
pub enum Question {
  MultipleChoice(MultipleChoice),
  ShortAnswer(ShortAnswer),
  Tracing(Tracing),
}

type IntermediateSchemas = Vec<Result<(QuizName, (i64, u64, CommitHash, Intern<Schema>)), usize>>;

impl Quizzes {
  fn load_schemas(answers: Vec<(&QuizName, &CommitHash)>) -> Result<IntermediateSchemas> {
    let rust_book_dir = &Path::new(RUST_BOOK_DIR).canonicalize()?;

    let answers_copy = answers.into_iter().enumerate().collect::<Vec<_>>();

    let load_schema =
      |repo: &mut git2::Repository,
       (index, (name, commit_hash)): (usize, (&QuizName, &CommitHash))| {
        let schema = {
          let obj_spec = format!("{commit_hash}:quizzes/{name}.toml");

          // Mini-hack: ran into some impossible data where quizzes were coming from a commit hash
          // when they didn't exist. Probably a stale telemetry script? For now, just filtering
          // those data points.
          let Ok(obj) = repo.revparse_single(&obj_spec) else {
          return Ok(Err(index))
        };

          let blob = obj.peel_to_blob()?;
          let s = String::from_utf8_lossy(blob.content());
          toml::from_str::<Schema>(&s).with_context(|| s.to_string())?
        };

        let schema = Intern::new(schema);

        let time = {
          let obj = repo.revparse_single(commit_hash)?;
          let commit = obj.peel_to_commit()?;
          commit.time().seconds()
        };

        let content_hash = {
          let mut hasher = DefaultHasher::new();
          schema.hash(&mut hasher);
          hasher.finish()
        };

        Ok(Ok((*name, (time, content_hash, *commit_hash, schema))))
      };

    answers_copy
      .into_par_iter()
      .progress_with_style(crate::pb_style())
      .map_init(
        || git2::Repository::open(rust_book_dir).expect("Failed to initialize git repo"),
        load_schema,
      )
      .collect()
  }

  fn index_schemas(schemas: IntermediateSchemas) -> Result<(Self, HashSet<usize>)> {
    let (ignored, schemas): (Vec<_>, Vec<_>) = schemas.into_iter().partition(|res| res.is_err());
    let ignored_inputs = ignored
      .into_iter()
      .map(|res| res.unwrap_err())
      .collect::<HashSet<_>>();

    let groups = schemas
      .into_iter()
      .map(|schema| schema.unwrap())
      .into_group_map();
    let mut schemas = HashMap::default();
    let mut commits: Commits = HashMap::default();
    for (name, mut v) in groups {
      v.sort_by_key(|(time, ..)| *time);

      for (_, content_hash, commit_hash, schema) in v {
        let new_schema = !schemas.contains_key(&content_hash);
        if new_schema {
          schemas.insert(content_hash, schema);
        }

        let quiz_commits = commits.entry(name).or_default();
        let last_version = quiz_commits
          .values()
          .map(|(_, version)| *version)
          .max()
          .unwrap_or(0);
        let version = if new_schema {
          last_version + 1
        } else {
          last_version
        };
        quiz_commits.insert(commit_hash, (content_hash, version));
      }
    }

    let all_lengths = commits
      .values()
      .flat_map(|versions| {
        versions.values().flat_map(|(hash, version)| {
          schemas[hash]
            .questions
            .iter()
            .filter_map(|q| {
              Some(match q {
                Question::MultipleChoice(MultipleChoice::MultipleChoiceNew(q)) => {
                  (q.id?, token::count_md_tokens(&q.prompt.prompt).ok()?)
                }
                Question::MultipleChoice(MultipleChoice::MultipleChoiceOld(q)) => {
                  (q.id?, token::count_md_tokens(&q.prompt.prompt).ok()?)
                }
                Question::ShortAnswer(q) => (q.id?, token::count_md_tokens(&q.prompt.prompt).ok()?),
                Question::Tracing(q) => (q.id?, token::count_rust_tokens(&q.prompt.program).ok()?),
              })
            })
            .map(|(id, len)| (id, (len, *version)))
        })
      })
      .into_group_map();

    let lengths = all_lengths
      .into_iter()
      .map(|(id, versions)| {
        let (len, _) = versions.into_iter().max_by_key(|(_, v)| *v).unwrap();
        (id, len)
      })
      .collect::<HashMap<_, _>>();

    Ok((
      Quizzes {
        schemas,
        commits,
        lengths,
      },
      ignored_inputs,
    ))
  }

  pub fn build(answers: Vec<(&QuizName, &CommitHash)>) -> Result<(Self, HashSet<usize>)> {
    let schemas = time!("load_schemas", Self::load_schemas(answers)?);
    time!("index_schemas", Self::index_schemas(schemas))
  }

  pub fn version(&self, quiz_name: &QuizName, commit_hash: &CommitHash) -> Result<usize> {
    let (_, version) = &self.commits[quiz_name]
      .get(commit_hash)
      .ok_or_else(|| anyhow!("Missing {quiz_name} at {commit_hash}"))?;
    Ok(*version)
  }

  pub fn schema_ref<'a>(&'a self, quiz_name: &QuizName, commit_hash: &CommitHash) -> &'a Schema {
    let (content_hash, _) = &self.commits[quiz_name][commit_hash];
    &self.schemas[content_hash]
  }
}

#[pymethods]
impl Quizzes {
  fn schema(&self, py: Python, quiz_name: &str, commit_hash: &str) -> PyResult<PyObject> {
    Ok(pythonize::pythonize(
      py,
      self.schema_ref(
        &ArrayString::from(quiz_name).unwrap(),
        &ArrayString::from(commit_hash).unwrap(),
      ),
    )?)
  }

  fn length(&self, question_id: &str) -> PyResult<usize> {
    let qid = QuestionId::new(question_id)?;
    Ok(self.lengths[&qid])
  }
}
