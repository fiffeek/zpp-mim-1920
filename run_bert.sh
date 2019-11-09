#!/usr/bin/env bash
SESSION="bert_pretraining"
wemux new-s -d -s "$SESSION"
wemux send-keys -t "$SESSION:0" "python3 colab.py" Enter
