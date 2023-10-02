#!/bin/zsh

source $HOME/.zshrc

set -e

gssh exp-core --command="sqlite3 rust-book-backend/server/log.sqlite '.backup /tmp/log.sqlite'"
gscp exp-core:/tmp/log.sqlite data/log.sqlite
