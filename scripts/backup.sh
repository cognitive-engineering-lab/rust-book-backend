#!/bin/bash

sqlite3 $HOME/rust-book-backend/server/log.sqlite ".backup ${HOME}/backups/log.backup-$(date +%Y-%m-%d).sqlite"
