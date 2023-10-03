use std::{
  fs::File,
  io::{BufWriter, Write},
};

use anyhow::Result;
use fxhash::FxHashMap as HashMap;
use itertools::Itertools;
use serde::Serialize;
use utils::answers::{QuestionId, SessionId};

#[derive(Serialize)]
struct IrtLine {
  subject_id: SessionId,
  responses: HashMap<QuestionId, u16>,
}

fn main() -> Result<()> {
  let (_quizzes, mut answers_flat) = utils::answers::load_answers_rs()?;
  answers_flat.retain(|ans| ans.question_id.is_some());

  let groups = answers_flat
    .into_iter()
    .map(|ans| (ans.session_id, (ans.question_id.unwrap(), ans.correct)))
    .into_group_map();

  let lines = groups
    .into_iter()
    .map(|(subject_id, responses)| IrtLine {
      subject_id,
      responses: responses.into_iter().collect::<HashMap<_, _>>(),
    })
    .collect_vec();

  let mut f = File::create("rbe.jsonlines")?;
  let mut writer = BufWriter::new(&mut f);
  for line in lines {
    serde_json::to_writer(&mut writer, &line)?;
    writer.write_all(b"\n")?;
  }
  writer.flush()?;

  Ok(())
}
