// pythonize = "0.17"
// rayon = {version = "1", default-features = false}
// git2 = {version = "0.15", default-features = false}
// anyhow = "1"
// toml = "0.5"
// thread_local = "1.1"
// itertools = "0.10"

use anyhow::{Context, Result};
use itertools::Itertools;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::path::Path;
use thread_local::ThreadLocal;

#[pyclass]
struct Quizzes {
  schemas: HashMap<u64, PyObject>,
  commits: HashMap<(String, String), (u64, usize)>,
}

const RUST_BOOK_DIR: &str = "../../rust-book";

impl Quizzes {
  fn build(py: Python, answers: Vec<(String, String)>) -> Result<Self> {
    let rust_book_dir = &Path::new(RUST_BOOK_DIR).canonicalize()?;

    let schemas = {
      let repo_tl: ThreadLocal<git2::Repository> = ThreadLocal::new();
      let mut answers_copy = answers.clone();
      answers_copy.dedup();

      let load_schema = |(name, commit_hash): (String, String)| {
        let repo = repo_tl
          .get_or(|| git2::Repository::open(rust_book_dir).expect("Failed to initialize git repo"));

        let schema = {
          let obj_spec = format!("{commit_hash}:quizzes/{name}.toml");
          let obj = repo
            .revparse_single(&obj_spec)
            .with_context(|| format!("Failed to parse objspec: {obj_spec}"))?;
          let blob = obj.peel_to_blob()?;
          String::from_utf8_lossy(blob.content()).parse::<toml::Value>()?
        };

        let time = {
          let obj = repo.revparse_single(&commit_hash)?;
          let commit = obj.peel_to_commit()?;
          commit.time().seconds()
        };

        let content_hash = {
          let questions = schema.as_table().expect("Schema")["questions"]
            .as_array()
            .expect("Schema");
          let cleaned = questions
            .iter()
            .map(|q| {
              let q = q.as_table().expect("Schema");
              ["id", "type", "prompt", "answer"]
                .iter()
                .filter_map(|k| q.get(*k))
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
          let mut hasher = DefaultHasher::new();
          // TODO: faster way to hash this than stringifying it?
          format!("{cleaned:?}").hash(&mut hasher);
          hasher.finish()
        };

        Ok((name, (time, content_hash, commit_hash, schema)))
      };

      answers_copy
        .into_par_iter()
        .map(load_schema)
        .collect::<Result<Vec<_>>>()?
    };

    let groups = schemas.into_iter().into_group_map();
    let mut schemas = HashMap::new();
    let mut commits = HashMap::new();
    for (name, mut v) in groups {
      v.sort_by_key(|(time, ..)| *time);

      for (_, content_hash, commit_hash, schema) in v {
        if !schemas.contains_key(&content_hash) {
          schemas.insert(content_hash, pythonize::pythonize(py, &schema)?);
        }
        let version = schemas.len() - 1;
        commits.insert((name.clone(), commit_hash), (content_hash, version));
      }
    }

    Ok(Quizzes { schemas, commits })
  }
}

#[pymethods]
impl Quizzes {
  #[new]
  fn py_new(py: Python, answers: Vec<(String, String)>) -> PyResult<Self> {
    Self::build(py, answers).map_err(|e| PyException::new_err(format!("{e:?}")))
  }

  fn schema(&self, py: Python, quiz_name: String, commit_hash: String) -> PyResult<PyObject> {
    let (content_hash, _) = &self.commits[&(quiz_name, commit_hash)];
    let schema = Py::clone_ref(&self.schemas[&content_hash], py);
    Ok(schema)
  }

  fn version(&self, quiz_name: String, commit_hash: String) -> PyResult<usize> {
    let (_, version) = &self.commits[&(quiz_name, commit_hash)];
    Ok(*version)
  }
}

#[pymodule]
fn rs_utils(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_class::<Quizzes>()?;
  Ok(())
}
