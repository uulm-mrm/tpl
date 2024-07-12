#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# fix user home directory permissions
chown -R "$TARGET_USER:$TARGET_USER" /home/
# fix tty premissions
chown -R "$TARGET_USER:$TARGET_USER" /dev/pts/
# link shell script
ln -s /etc/zsh/zshrc /home/$TARGET_USER/.zshrc >/dev/null 2>&1;
ln -s /etc/tmux.conf /home/$TARGET_USER/.tmux.conf >/dev/null 2>&1;
# hand over to target user
su -m $TARGET_USER
