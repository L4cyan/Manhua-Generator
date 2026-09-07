# Cloud jobs

Batch work pushed to Kaggle with the API: fire it off, walk away, collect a
file. That is what `kaggle kernels push` is actually good at — unlike a
long-lived tunnel server, which is better held open interactively.

    train_lora/     character identity LoRA training

The TLS note in `manhua/net.py` applies here too: the `kaggle` CLI binary
fails on machines with antivirus HTTPS inspection. `manhua/cloud/kaggle.py`
drives the Python API directly with `truststore` injected first, which is why
`manhua train` works where the raw CLI does not.
