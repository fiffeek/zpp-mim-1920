import os
import logging


tmux_wrapper = "wemux"


class TmuxSession:
    def __init__(self, session_name, detached=True):
        self.session_name = session_name
        self.detached = detached
        logging.info(f"{tmux_wrapper} Console instantiated, name=[{session_name}]")

    def run_wemux_session(self):
        if self.detached:
            os.system(f"{tmux_wrapper} new-s -d -s \"{self.session_name}\"")
        else:
            os.system(f"{tmux_wrapper} new-s -s \"{self.session_name}\"")
        logging.info(f"Console name=[{self.session_name}] is turned on and detached=[{self.detached}]")
        self.info_for_user()

    def send_keys(self, command, session_number=0):
        os.system(f"{tmux_wrapper} send-keys -t \"{self.session_name}:{session_number}\" \"{command}\" Enter")

    def info_for_user(self):
        logging.info(f"To connect to session:")
        logging.info(f" $ wemux attach-session -t {self.session_name}")
        logging.info(f"To kill the session:")
        logging.info(f" $ wemux kill-session -t {self.session_name}")
        logging.info(f"To detach from session:")
        logging.info(f" $ ctrl + b + [wait some time] + d")
