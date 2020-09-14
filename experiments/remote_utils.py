import os
import torch
import paramiko
import datetime
import traceback
import functools
import socket
import yagmail
from scp import SCPClient
from paramiko import SSHClient
from transformers import trainer_utils
from typing import List, Optional, Any, Tuple, Dict

from experiments import misc_utils
from experiments import constants

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_and_verify_environment() -> None:
    # Check the environment
    if not os.path.exists(constants.REMOTE_DEFAULT_SSH_KEY_FILENAME):
        raise ValueError(f"{constants.REMOTE_DEFAULT_SSH_KEY_FILENAME} does not exist")

    if constants.REMOTE_DEFAULT_REMOTE_BASE_DIR is None:
        raise ValueError(f"`REMOTE_BASE_DIR` is not set.")

    if trainer_utils.is_wandb_available() is False:
        raise ValueError("Weight And Bias is not set.")


class EmailClient(object):
    def __init__(self,
                 sender_email: Optional[str] = None,
                 sender_password: Optional[str] = None,
                 recipient_email: Optional[str] = None) -> None:

        if sender_email is None:
            sender_email = constants.REMOTE_DEFAULT_SENDER_EMAIL

        if sender_password is None:
            sender_password = constants.REMOTE_DEFAULT_SENDER_PASSWORD

        if recipient_email is None:
            recipient_email = constants.REMOTE_DEFAULT_RECIPIENT_EMAIL

        self._sender_email = sender_email
        self._recipient_email = recipient_email
        self._sender = yagmail.SMTP(sender_email)

    def send(self, subject: str, contents: str) -> List[str]:
        # Prepare Headers
        message_time = datetime.datetime.now()
        message_time.strftime(DATE_FORMAT)
        host_name = socket.gethostname()
        # Prepare Contents
        full_contents = [f"Time: {message_time} | Host: {host_name}", contents]

        # Send!
        self._sender.send(self._recipient_email, subject, full_contents)
        return full_contents

    def close(self) -> None:
        self._sender.close()


def send_email(subject: str,
               contents: str,
               sender_email: Optional[str] = None,
               sender_password: Optional[str] = None,
               recipient_email: Optional[str] = None) -> List[str]:
    if sender_email is None:
        sender_email = constants.REMOTE_DEFAULT_SENDER_EMAIL

    if sender_password is None:
        sender_password = constants.REMOTE_DEFAULT_SENDER_PASSWORD

    if recipient_email is None:
        recipient_email = constants.REMOTE_DEFAULT_RECIPIENT_EMAIL

    # Prepare Headers
    message_time = datetime.datetime.now()
    message_time.strftime(DATE_FORMAT)
    host_name = socket.gethostname()
    # Prepare Contents
    full_contents = [f"Time: {message_time} | Host: {host_name}", contents]

    with yagmail.SMTP(sender_email) as sender:
        sender.send(recipient_email, subject, full_contents)

    return full_contents


class ScpClient(object):
    def __init__(self,
                 server_address: Optional[str] = None,
                 server_username: Optional[str] = None,
                 server_password: Optional[str] = None,
                 ssh_key_filename: Optional[str] = None) -> None:
        if server_address is None:
            server_address = constants.REMOTE_DEFAULT_SERVER_ADDRESS
        if server_username is None:
            server_username = constants.REMOTE_DEFAULT_SERVER_USERNAME
        if server_password is None:
            server_password = constants.REMOTE_DEFAULT_SERVER_PASSWORD
        if ssh_key_filename is None:
            ssh_key_filename = constants.REMOTE_DEFAULT_SSH_KEY_FILENAME

        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())
        ssh.connect(hostname=server_address,
                    username=server_username,
                    key_filename=ssh_key_filename)
        self._ssh = ssh

    def scp_file_to_remote(
            self,
            local_file_name: str,
            remote_file_name: str,
            recursive: bool = False) -> None:
        with SCPClient(self._ssh.get_transport()) as scp:
            scp.put(files=local_file_name,
                    remote_path=remote_file_name,
                    recursive=recursive)

    def scp_file_from_remote(
            self,
            local_file_name: str,
            remote_file_name: str,
            recursive: bool = False) -> None:
        with SCPClient(self._ssh.get_transport()) as scp:
            scp.get(remote_path=remote_file_name,
                    local_path=local_file_name,
                    recursive=recursive)

    def save_and_mirror_scp_to_remote(
            self,
            object_to_save: Any,
            local_file_name: str,
            remote_file_name: str) -> None:

        torch.save(object_to_save, local_file_name)
        self.scp_file_to_remote(
            local_file_name=local_file_name,
            remote_file_name=remote_file_name)


def save_and_mirror_scp_to_remote(
        object_to_save: Any,
        file_name: str,
        server_address: Optional[str] = None,
        server_username: Optional[str] = None,
        server_password: Optional[str] = None,
        ssh_key_filename: Optional[str] = None) -> Tuple[ScpClient, str]:

    client = ScpClient(
        server_address=server_address,
        server_username=server_username,
        server_password=server_password,
        ssh_key_filename=ssh_key_filename)

    host_name = socket.gethostname()
    remote_file_name = f"{file_name}.{host_name}"
    if constants.REMOTE_DEFAULT_REMOTE_BASE_DIR is not None:
        remote_file_name = os.path.join(
            constants.REMOTE_DEFAULT_REMOTE_BASE_DIR,
            remote_file_name)

    client.save_and_mirror_scp_to_remote(
        object_to_save=object_to_save,
        local_file_name=file_name,
        remote_file_name=remote_file_name)

    return client, remote_file_name


def maybe_download_file_from_remote_server(
        local_file_name: str,
        remote_file_name: Optional[str] = None,
        server_address: Optional[str] = None,
        server_username: Optional[str] = None,
        server_password: Optional[str] = None,
        ssh_key_filename: Optional[str] = None
) -> None:
    """Check if `local_file_name` exists, or scp from `remote_file_name` if not."""
    if os.path.exists(local_file_name):
        print("Local file exists")
        return

    print("Local file does not exist, syncing from remote server")
    if remote_file_name is None:
        raise ValueError

    client = ScpClient(
        server_address=server_address,
        server_username=server_username,
        server_password=server_password,
        ssh_key_filename=ssh_key_filename)

    client.scp_file_from_remote(
        local_file_name=local_file_name,
        remote_file_name=remote_file_name)


def load_file_from_local_or_remote(
        local_file_name: str,
        remote_file_name: Optional[str] = None,
        server_address: Optional[str] = None,
        server_username: Optional[str] = None,
        server_password: Optional[str] = None,
        ssh_key_filename: Optional[str] = None
) -> Any:
    """Load `local_file_name` if exists, or download + load
       `remote_file_name` to `local_file_name` if not
    """
    maybe_download_file_from_remote_server(
        local_file_name=local_file_name,
        remote_file_name=remote_file_name)

    return torch.load(local_file_name)


def test_save_and_mirror_scp_to_remote():
    import torch  # Only importing torch here
    tensor = torch.rand(100, 100)
    client, remote_file_name = save_and_mirror_scp_to_remote(
        object_to_save=tensor,
        file_name="random_tensor_test.pt")

    client.scp_file_from_remote(
        remote_file_name=remote_file_name,
        local_file_name="fetched_random_tensor_test.pt")

    fetched_tensor = torch.load("fetched_random_tensor_test.pt")
    misc_utils.remove_file_if_exists("random_tensor_test.pt")
    misc_utils.remove_file_if_exists("fetched_random_tensor_test.pt")
    print(f"Remote and Local Matched: {(fetched_tensor == tensor).all()}")
