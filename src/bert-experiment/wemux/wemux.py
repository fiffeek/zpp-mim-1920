import os
import logging


tmux_wrapper = "wemux"


class TmuxSession:
    def __init__(self, session_name, detached=True):
        self.session_name = session_name
        self.detached = detached
        logging.info("{} Console instantiated, name=[{}]".format(tmux_wrapper, session_name))

    def run_wemux_session(self):
        if self.detached:
            os.system("{} new-s -d -s \"{}\"".format(tmux_wrapper, self.session_name))
        else:
            os.system("{} new-s -s \"{}\"".format(tmux_wrapper, self.session_name))
        logging.info("Console name=[{}] is turned on and detached=[{}]".format(self.session_name, self.detached))
        self.info_for_user()

    def send_keys(self, command, session_number=0):
        os.system("{} send-keys -t \"{}:{}\" \"{}\" Enter"
                  .format(tmux_wrapper, self.session_name, session_number, command))

    def connect_to_session(self):
        os.system(" $ {} attach-session -t {}".format(tmux_wrapper, self.session_name))

    def info_for_user(self):
        logging.info("To connect to session:")
        logging.info(" $ {} attach-session -t {}".format(tmux_wrapper, self.session_name))
        logging.info("To kill the session:")
        logging.info(" $ {} kill-session -t {}".format(tmux_wrapper, self.session_name))
        logging.info("To detach from session:")
        logging.info(" $ ctrl + b + [wait some time] + d")
