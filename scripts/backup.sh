#!/bin/bash
cd $HOME/rust-book-backend/server
tar -czpf $HOME/backups/rust-book-logs-$(date +%d-%m-%Y).tar.gz *.log
