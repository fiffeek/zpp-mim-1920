import os
import argparse

ip_address_cmd = "gcloud compute addresses list --format \"value(address_range())\""
username_cmd = "gcloud config list account --format \"value(core.account)\""
username = os.popen(username_cmd).read().split("@")[0]
ip = os.popen(ip_address_cmd).read().replace("\n", "")

parser = argparse.ArgumentParser()
parser.add_argument("--local-folder", default="vm", help="Name of a local folder to mount")
parser.add_argument("--vm-ip", default=f"{ip}", help="Ip of a vm on gcloud")
parser.add_argument("--gcloud-username", default=f"{username}", help="Ip of a vm on gcloud")
parser.add_argument("--home-dir", default=f"{os.environ['HOME']}", help="Home dir, so that it can find identity")
args = parser.parse_args()

os.system(f"mkdir {args.local_folder}")
os.system(f"sshfs -ononempty, {args.gcloud_username}@{args.vm_ip}:/home {args.local_folder} -o "
          f"IdentityFile=\"{args.home_dir}/.ssh/id_rsa\" -o auto_cache -o cache_timeout=115200 "
          "-o attr_timeout=115200 -o entry_timeout=1200 -o max_readahead=90000 -o large_read -o big_writes "
          "-o no_remote_lock -o Compression=no -o kernel_cache ")
