# Data layout on the cluster

The profile mirror copies `/home/<user>` from node_a to node_b every 15 minutes.
It exists so that a person's configuration follows them to the other node. It is
not a data channel. A large file under a home directory reaches the peer within
15 minutes and fills its disk.

## Rules

| what | where | mirrored |
| --- | --- | --- |
| configuration, dotfiles, agent state, small project trees | `/home/<user>` | yes, by `profile-mirror.timer` (15 min, no delete) |
| model weights that every node serves | `/models/shared` | yes, by `models-mirror.timer` (10 min) |
| datasets, extracted features, training checkpoints, per-person weights | `/models/<user>` (`$SPARK_MODELS_MINE`) | no |

- Put every file above 1 GiB under `/models/shared` or `/models/<user>`. The
  profile mirror skips files above 1 GiB, feature trees (`feats*/`), `*.safetensors`
  and `*.gguf`, so a mistake does not fill the peer. The file then exists on one
  node only.
- A project tree that needs a large directory keeps a symbolic link in the tree that
  points into `/models/<user>` or `/models/shared`. The mirror copies the link, not
  the target.
- A container that reads or writes `/models` must mount it (`-v /models:/models`).
  A symbolic link into `/models` does not resolve inside a container that mounts
  only `/home/<user>`.
- Copy a large file to the peer on purpose: `rsync -a <file> <peer>:/models/<user>/...`.

## History

2026-09-19: the mirror had copied 1.4 TB of training features and checkpoints to
node_b as full duplicates and was in the middle of a 0.9 TB copy when it was
stopped. The guards above and the `/models/<user>` rule date from that day.
