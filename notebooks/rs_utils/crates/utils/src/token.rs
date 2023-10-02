use anyhow::{anyhow, Result};
use markdown::mdast::Node;
use proc_macro2::{TokenStream, TokenTree};
use std::str::FromStr;

pub fn count_rust_tokens(s: &str) -> Result<usize> {
  let stream = TokenStream::from_str(s).map_err(|e| anyhow!("{e}\n{s}"))?;

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

pub fn count_md_tokens(s: &str) -> Result<usize> {
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

  fn default_count(s: &str) -> usize {
    s.split(' ').count()
  }

  for node in nodes {
    let n = match node {
      Node::Text(text) => default_count(&text.value),
      Node::Code(code) => match code.lang.as_deref() {
        None | Some("ide") | Some("rust") => {
          count_rust_tokens(&code.value).unwrap_or_else(|_| default_count(&code.value))
        }
        _ => default_count(&code.value),
      },
      _ => 0,
    };
    count += n;
  }
  Ok(count)
}

#[test]
fn test_count_md_tokens() {
  let pairs = vec![
    ("Hello world", 2),
    ("*Test wo*rld", 3),
    (
      r#"
```
fn main() {}
```    
"#,
      6,
    ),
    (
      r#"
```text
Hello world 
```
"#,
      2,
    ),
  ];

  for (inp, outp) in pairs {
    assert_eq!(count_md_tokens(inp).unwrap(), outp, "{inp}");
  }
}
