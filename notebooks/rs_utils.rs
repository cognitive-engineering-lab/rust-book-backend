// pythonize = "0.17"
// rayon = {version = "1", default-features = false}
// git2 = {version = "0.15", default-features = false}
// anyhow = "1"
// toml = "0.5"
// thread_local = "1.1"
// itertools = "0.10"
// memory-stats = "1"
// serde = {version = "1", features = ["derive"]}
// internment = {version = "0.7.1", features = ["arena"]}
// indicatif = {version = "0.17", features = ["rayon"]}
// proc-macro2 = "1.0.67"
// markdown = "1.0.0-alpha.14"

use anyhow::{anyhow, Context, Result};
use indicatif::ParallelProgressIterator;
use internment::Arena;
use itertools::Itertools;
use markdown::mdast::Node;
use proc_macro2::{TokenStream, TokenTree};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::str::FromStr;
use thread_local::ThreadLocal;

type Commits = HashMap<String, HashMap<String, (u64, usize)>>;
#[pyclass]
struct Quizzes {
  schemas: HashMap<u64, PyObject>,
  commits: Commits,
  ignored_inputs: Vec<usize>,
}

const RUST_BOOK_DIR: &str = "../code/rust-book";

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct Schema {
  questions: Vec<Question>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
enum QuestionType {
  Tracing,
  MultipleChoice,
  ShortAnswer,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(untagged)]
enum MultipleChoice {
  MultipleChoiceNew(MultipleChoiceNew),
  MultipleChoiceOld(MultipleChoiceOld),
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct MultipleChoiceNew {
  id: Option<String>,
  prompt: MultipleChoicePrompt,
  answer: MultipleChoiceAnswer,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct MultipleChoicePrompt {
  prompt: String,
  distractors: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct MultipleChoiceAnswer {
  answer: MultipleChoiceAnswerOptions,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(untagged)]
enum MultipleChoiceAnswerOptions {
  Single(String),
  Multiple(Vec<String>),
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct MultipleChoiceOld {
  id: Option<String>,
  prompt: MultipleChoiceOldPrompt,
  answer: MultipleChoiceOldAnswer,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct MultipleChoiceOldPrompt {
  prompt: String,
  choices: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct MultipleChoiceOldAnswer {
  answer: usize,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct ShortAnswer {
  id: Option<String>,
  prompt: ShortAnswerPrompt,
  answer: ShortAnswerAnswer,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct ShortAnswerPrompt {
  prompt: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct ShortAnswerAnswer {
  answer: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct Tracing {
  id: Option<String>,
  prompt: TracingPrompt,
  answer: TracingAnswer,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
struct TracingPrompt {
  program: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(untagged)]
enum TracingAnswer {
  DoesNotCompile {
    #[serde(rename = "doesCompile")]
    does_compile: bool,
    #[serde(rename = "lineNumber")]
    line_number: usize,
  },
  DoesCompile {
    #[serde(rename = "doesCompile")]
    does_compile: bool,
    stdout: String,
  },
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(tag = "type")]
enum Question {
  MultipleChoice(MultipleChoice),
  ShortAnswer(ShortAnswer),
  Tracing(Tracing),
}

impl Quizzes {
  fn build(py: Python, answers: Vec<(String, String)>) -> Result<Self> {
    let rust_book_dir = &Path::new(RUST_BOOK_DIR).canonicalize()?;

    let schema_arena: Arena<Schema> = Arena::new();

    let schemas = {
      let repo_tl: ThreadLocal<git2::Repository> = ThreadLocal::new();
      let answers_copy = answers.into_iter().enumerate().collect::<Vec<_>>();

      let load_schema = |(index, (name, commit_hash)): (usize, (String, String))| {
        let repo = repo_tl
          .get_or(|| git2::Repository::open(rust_book_dir).expect("Failed to initialize git repo"));

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
          toml::from_str::<Schema>(&*s).with_context(|| s.to_string())?
        };

        let schema = schema_arena.intern(schema);

        let time = {
          let obj = repo.revparse_single(&commit_hash)?;
          let commit = obj.peel_to_commit()?;
          commit.time().seconds()
        };

        let content_hash = {
          let mut hasher = DefaultHasher::new();
          schema.hash(&mut hasher);
          hasher.finish()
        };

        Ok(Ok((name, (time, content_hash, commit_hash, schema))))
      };

      answers_copy
        .into_par_iter()
        .progress()
        .map(load_schema)
        .collect::<Result<Vec<_>>>()?
    };

    let (ignored, schemas): (Vec<_>, Vec<_>) = schemas.into_iter().partition(|res| res.is_err());
    let ignored_inputs = ignored
      .into_iter()
      .map(|res| res.unwrap_err())
      .collect::<Vec<_>>();

    if let Some(usage) = memory_stats::memory_stats() {
      println!(
        "Current physical memory usage: {} MB",
        usage.physical_mem / 1024 / 1024
      );
    }

    let groups = schemas
      .into_iter()
      .map(|schema| schema.unwrap())
      .into_group_map();
    let mut schemas = HashMap::new();
    let mut commits: Commits = HashMap::new();
    for (name, mut v) in groups {
      v.sort_by_key(|(time, ..)| *time);

      for (_, content_hash, commit_hash, schema) in v {
        let new_schema = !schemas.contains_key(&content_hash);
        if new_schema {
          schemas.insert(content_hash, pythonize::pythonize(py, &*schema)?);
        }

        let quiz_commits = commits.entry(name.clone()).or_default();
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

    if let Some(usage) = memory_stats::memory_stats() {
      println!(
        "Current physical memory usage: {} MB",
        usage.physical_mem / 1024 / 1024
      );
    }

    Ok(Quizzes {
      schemas,
      commits,
      ignored_inputs,
    })
  }
}

#[pymethods]
impl Quizzes {
  #[new]
  fn py_new(py: Python, answers: Vec<(String, String)>) -> PyResult<Self> {
    Self::build(py, answers).map_err(|e| PyException::new_err(format!("{e:?}")))
  }

  fn schema(&self, py: Python, quiz_name: &str, commit_hash: &str) -> PyObject {
    let (content_hash, _) = &self.commits[quiz_name][commit_hash];
    Py::clone_ref(&self.schemas[&content_hash], py)
  }

  fn version(&self, quiz_name: &str, commit_hash: &str) -> PyResult<usize> {
    let (_, version) = &self.commits[quiz_name]
      .get(commit_hash)
      .ok_or_else(|| PyException::new_err(format!("Missing {quiz_name} at {commit_hash}")))?;
    Ok(*version)
  }

  fn ignored_inputs(&self) -> Vec<usize> {
    self.ignored_inputs.clone()
  }
}

fn count_rust_tokens(s: &str) -> Result<usize> {
  let stream = TokenStream::from_str(s).map_err(|e| anyhow!("{e}"))?;

  fn count(stream: TokenStream) -> usize {
    stream
      .into_iter()
      .map(|tree| match tree {
        TokenTree::Group(g) => count(g.stream()) + 2,
        _ => 1,
      })
      .sum()
  }

  Ok(count(stream))
}

#[pyfunction]
fn count_md_tokens(s: &str) -> PyResult<usize> {
  let root =
    markdown::to_mdast(s, &markdown::ParseOptions::default()).map_err(|s| anyhow!("{s}"))?;

  fn collect_nodes(root: &Node) -> Vec<&Node> {
    let mut queue = vec![root];
    let mut nodes = vec![];
    while let Some(node) = queue.pop() {
      nodes.push(node);
      if let Some(children) = node.children() {
        queue.extend(children);
      }
    }
    nodes
  }

  let nodes = collect_nodes(&root);
  let mut count = 0;
  for node in nodes {
    let n = match node {
      Node::Text(text) => text.value.split(" ").count(),
      Node::Code(code) => count_rust_tokens(&code.value)?,
      _ => 0,
    };
    count += n;
  }
  Ok(count)
}

#[pymodule]
fn rs_utils(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_class::<Quizzes>()?;
  m.add_function(&count_md_tokens)?;
  Ok(())
}
