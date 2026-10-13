#!/bin/bash
# Install the server script and a per-user LaunchAgent that starts it at login and restarts it if it fails.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p ~/.local/bin ~/Library/LaunchAgents ~/Library/Logs
install -m 755 "$HERE/bonsai-ane-serve.sh" ~/.local/bin/bonsai-ane-serve.sh
sed "s#__HOME__#$HOME#g" "$HERE/com.user.bonsai-ane.plist.template" > ~/Library/LaunchAgents/com.user.bonsai-ane.plist
plutil -lint ~/Library/LaunchAgents/com.user.bonsai-ane.plist
launchctl bootout gui/$(id -u)/com.user.bonsai-ane 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.bonsai-ane.plist
echo "Started. Log: ~/Library/Logs/bonsai-ane.log. Health: curl -s http://127.0.0.1:8765/health"
