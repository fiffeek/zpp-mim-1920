SESSION="session1"
tmux new-s -d -s "$SESSION"
tmux send-keys -t "$SESSION:0" "python3 colab.py" Enter