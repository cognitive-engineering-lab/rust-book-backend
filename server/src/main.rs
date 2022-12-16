#![feature(decl_macro)]
use anyhow::{bail, Result};
use flate2::{write::ZlibEncoder, Compression};
use std::io::{Read, Write};

fn encode(buf: Vec<u8>) -> Result<Vec<u8>> {
  let mut enc = ZlibEncoder::new(Vec::new(), Compression::best());
  enc.write_all(&buf)?;
  Ok(enc.finish()?)
}

#[rocket_contrib::database("logs")]
struct Logs(rocket_contrib::databases::rusqlite::Connection);

const TABLES: &[&str] = &["answers", "bug", "feedback", "runtime_error"];

#[rocket::post("/logs/<table>", format = "json", data = "<data>")]
fn log(logs: Logs, table: String, data: rocket::Data) -> Result<&'static str> {
  if !TABLES.contains(&table.as_str()) {
    bail!("Invalid table: {table}");
  }

  let mut buf = Vec::new();
  data.open().read_to_end(&mut buf)?;
  let encoded = encode(buf)?;

  let cmd = format!("INSERT INTO {table} (data) VALUES (?1)");
  logs.execute(&cmd, &[&encoded])?;
  Ok("success")
}

#[rocket::get("/")]
fn index() -> &'static str {
  "MIND OVER COMPUTER"
}

fn main() {
  let r = rocket::ignite()
    .attach(rocket_cors::Cors::from_options(&Default::default()).unwrap())
    .attach(Logs::fairing())
    .mount("/", rocket::routes![index, log]);

  let logs = Logs::get_one(&r).unwrap();
  for t in TABLES {
    let cmd = format!("CREATE TABLE IF NOT EXISTS {t} (id INTEGER PRIMARY KEY, data BLOB)");
    logs.execute(&cmd, &[]).unwrap();
  }

  r.launch();
}
