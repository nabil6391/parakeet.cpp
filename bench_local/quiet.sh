#!/bin/bash
# quiet.sh pause|resume : SIGSTOP/SIGCONT background test browsers, their node test runners, and vite dev servers
PIDS_FILE=/tmp/pk_quiet_pids
if [ "$1" = pause ]; then
  pgrep -f "Chrome for Testing" > $PIDS_FILE
  ps -Ao pid,command | awk '/node .*test\/[a-z_-]+\.mjs/ && !/awk/ {print $1}' >> $PIDS_FILE
  pgrep -f "node_modules/.bin/vite" >> $PIDS_FILE
  pgrep -f "scripts/pipeline.ts" >> $PIDS_FILE          # job launchers + their workers
  pgrep -f "node test/.*\.mjs" >> $PIDS_FILE
  sort -u $PIDS_FILE -o $PIDS_FILE
  xargs kill -STOP < $PIDS_FILE 2>/dev/null; echo "paused $(wc -l < $PIDS_FILE | tr -d ' ') processes"
else
  [ -f $PIDS_FILE ] && xargs kill -CONT < $PIDS_FILE 2>/dev/null; echo "resumed $(wc -l < $PIDS_FILE 2>/dev/null | tr -d ' ') processes"; rm -f $PIDS_FILE
fi
